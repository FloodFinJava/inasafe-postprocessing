# coding=utf-8

"""Estimate financial losses using output from inasafe"""

import os
import re
import logging
from xml.etree import ElementTree
import warnings

import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import toml

CONF_FILE = "conf.toml"


class LossModel(object):
    """Estimate losses for a single scenario
    """
    def __init__(self, conf, loss_curves, styling_path, impact_map_path):
        """conf: a dictionnary
        """
        self.cat_ranges = {}
        self.styling = ElementTree.parse(styling_path).getroot()
        self.impacts_map = gpd.read_file(impact_map_path)
        self.asset_values = pd.read_csv(os.path.normpath(conf['input']['asset_values']), index_col='class',squeeze=True)
        self.impact_hazard_class = conf['input']['impact']['col_hazard_class']
        self.impact_hazard_id = conf['input']['impact']['col_hazard_id']
        self.impact_exposure_class = conf['input']['impact']['col_exposure_class']
        self.impact_agg_name = conf['input']['impact']['col_agg_name']
        self.impact_affected = conf['input']['impact']['col_affected']
        # Variables that change in T
        self.impact_T_var = [self.impact_hazard_class,
                             self.impact_hazard_id,
                             self.impact_affected,
                             'damages', 'exceed_loss']
        self.curve_max = conf['input']['loss_curves']['max_value']
        self.default_curve = conf['input']['loss_curves']['default']
        self.loss_curves = loss_curves
        return None

    def read_hazard_cats(self):
        """read qgis styling file to extract hazard category ranges
        """
        for fcat in self.styling.find('renderer-v2').find('categories'):
            fcat_str = fcat.attrib['label']
            # Category label
            cat_label_str = re.split(' <| >| \(', fcat_str)[0]
            # remove label from the original string
            fcat_wo_label = fcat_str[len(cat_label_str):]
            # remove the number of buildings and 'm'
            try:
                range_str = fcat_wo_label.split('(')[:-1][0].split('m')[:-1][0].strip()
            except IndexError:
                range_str = ''
            if range_str:
                # print(range_str)
                # determine low and high range
                low = 0.
                high = self.curve_max
                try:
                    sp = range_str.strip('> ').split(' - ')
                    high = float(sp[1])
                    low = float(sp[0])
                except IndexError:  # only one value
                    if range_str.startswith('> '):
                        low = float(range_str[1:])
                    elif range_str.startswith('<= '):
                        high = float(range_str[2:])
                self.cat_ranges[cat_label_str.lower()] = (low, high)
            else:
                self.cat_ranges[cat_label_str.lower()] = (np.nan, np.nan)
        return self

    def damage_by_category(self):
        """Assign a damage value to each hazard category
        """
        df_ranges = pd.DataFrame(self.cat_ranges).T
        df_ranges.columns = ['lower', 'upper']
        rg_mean = (df_ranges['lower'] + df_ranges['upper'])/2
        df_ranges['mean'] = rg_mean
        # Set level as index to facilitate concat of damage %
        rg_mean_ridx = rg_mean.reset_index()
        rg_mean_ridx.columns = ['cat', 'level']
        rg_mean_ridx = rg_mean_ridx.set_index('level').sort_index()
        for curve_name in self.loss_curves.columns:
            df_curve = self.loss_curves[curve_name]
            # find damage % for each category
            range_damage = df_curve.where(np.isin(df_curve.index, rg_mean)).dropna()
            # Assign damage to category
            rg_mean_ridx[curve_name] = range_damage
        self.damages = rg_mean_ridx.set_index('cat')
        return self

    def _damage_estimate(self, row):
        """Estimate % damage for a single row of attribute table.
        """
        damages_dict = self.damages[self.default_curve].to_dict()
        match_str = row[self.impact_hazard_class].replace('_', ' ')
        if match_str == 'not exposed':
            return 0
        else:
            return damages_dict[match_str]

    def estimate_damage(self):
        """Estimate the damage (percentage of loss)
        according to depth-damage (vulnerability) curve.
        """
        self.impacts_map['damages'] = self.impacts_map.apply(self._damage_estimate, axis=1)
        return self

    def estimate_losses(self):
        """Compute financial losses: damage ratio x building value.
        """
        self.impacts_map['exceed_loss'] = self.impacts_map['damages'] * self.impacts_map['building_value']
        return self

    def _compute_building_value(self, row, v_dict):
        """the 'size' column is the building surface area in m².
        v_dict is the building value/m² per building category.
        """
        return row['size'] * v_dict[row[self.impact_exposure_class]]

    def estimate_building_value(self):
        """Multiply the building surface area by the unit value.
        """
        values_dict = self.asset_values.to_dict()
        self.impacts_map['building_value'] = self.impacts_map.apply(self._compute_building_value, axis=1, v_dict=values_dict)
        return self

    def get_losses_by_area(self):
        # Sum losses by aggregation area
        return self.impacts_map.groupby(self.impact_agg_name)['exceed_loss'].sum()


class FinancialComputation(object):
    """Estimate losses for a series of scenarios
    """
    def __init__(self, conf_file):
        with open(conf_file, 'r') as toml_file:
            toml_string = toml_file.read()
            self.conf = toml.loads(toml_string)
        self._read_conf_file()
        self._read_vulnerability_curve()
        self.loss_model_dict = self._read_scenarios()

    def _read_conf_file(self):
        """Load configuration data from the toml conf file
        """

        self.basepath = os.path.normpath(self.conf['input']['base_path'])
        self.results_dir = os.path.normpath(self.conf['input']['impact']['base_path'])
        # depth-damage curves
        self.loss_curves_path = os.path.normpath(os.path.join(self.basepath,
                                                              self.conf['input']['loss_curves']['path']))
        self.curve_max = self.conf['input']['loss_curves']['max_value']
        self.curve_resol = self.conf['input']['loss_curves']['step']
        # impact map
        self.impact_styling_name = self.conf['input']['impact']['impact_styling_name']
        self.impact_map_name = self.conf['input']['impact']['impact_map_name']
        # Information about insurance premium calculation
        self.insur_rt_ec = self.conf['input']['insurance']['rt_ec']
        self.insur_rc = self.conf['input']['insurance']['rc']
        self.insur_margin = self.conf['input']['insurance']['margin']
        # Summary of losses by aggregation
        self.losses_summary = None
        # Geographical summary
        self.losses_ds = None
        # Output
        self.output_summary_path = os.path.join(self.basepath,
                                                self.conf['output']['summary_file'])
        self.output_summary_map_path = os.path.join(self.basepath,
                                                self.conf['output']['summary_map'])

    def _read_vulnerability_curve(self):
        """Read curves from CSV.
        Interpolate them to the given resolution
        """
        loss_curves = pd.read_csv(self.loss_curves_path, index_col=0)
        new_index = np.arange(self.curve_max, step=self.curve_resol)
        self.loss_curves = loss_curves.reindex(loss_curves.index.union(new_index))
        self.loss_curves.interpolate('index', inplace=True)
        return self

    def _read_scenarios(self):
        """Read each scenario generated by inasafe.
        The return period is infered from the dir name, ex. Q200.
        """
        loss_model_dict = {}
        for d in os.listdir(self.results_dir):
            # Infer return period in years
            re_match = re.search('Q\d*', d)
            abspath = os.path.normpath(os.path.join(self.results_dir, d))
            if os.path.isdir(abspath) and re_match:
                return_period = int(re_match.group()[1:])
                # Estimate losses for each scenario
                styling_path = os.path.normpath(os.path.join(abspath,
                                                             self.impact_styling_name))
                impact_map_path = os.path.normpath(os.path.join(abspath,
                                                                self.impact_map_name))
                try:
                    loss_model = LossModel(self.conf, self.loss_curves,
                                           styling_path, impact_map_path)
                except FileNotFoundError as e:
                    warnings.warn("File {} not found".format(abspath))
                    continue
                else:
                    print(d)
                    loss_model_dict[return_period] = loss_model
        return loss_model_dict

    def estimate_losses(self):
        """Iterate through all the scenario results and estimate damages and losses.
        Keep a summary of losses and a xarray DataSet of all scenarios.
        """
        losses_per_area_list = []
        loss_map_list = []
        for return_period, loss_model in self.loss_model_dict.items():
            loss_model.read_hazard_cats().damage_by_category()
            loss_model.estimate_building_value()
            loss_model.estimate_damage().estimate_losses()
            ds_losses = loss_model.impacts_map.to_xarray()
            # Only some variables vary in T
            for var in loss_model.impact_T_var:
                ds_losses[var] = ds_losses[var].expand_dims('T')
            ds_losses.coords['T'] = [return_period]
            loss_map_list.append(ds_losses)
            # summary
            losses_by_area = loss_model.get_losses_by_area().rename(return_period)
            losses_per_area_list.append(losses_by_area)
        self.losses_summary = pd.concat(losses_per_area_list, axis='columns')
        # Create the losses dataset
        self.losses_ds = xr.concat(loss_map_list, dim='T', data_vars='minimal').sortby('T')
        return self

    def expected_losses(self):
        """Transform exceedance probability to p(event).
        Estimate the expected loss next year per building.
        """
        self.losses_ds['exceed_prob'] = 1 / self.losses_ds['T']
        ds_list = []
        # Loop accross buildings to do the sorting by loss
        for building_idx in self.losses_ds['index']:
            losses = self.losses_ds[['exceed_loss', 'exceed_prob']].sel(index=building_idx).sortby('exceed_loss')
            losses = losses.expand_dims('index')
            losses.coords['index'] = [building_idx]
            # Event probability
            p_event = -losses['exceed_prob'].diff(dim='T', label='lower')
            p_event_last = losses['exceed_prob'].isel(T=-1)
            ds = xr.concat([p_event, p_event_last], dim='T').rename('event_prob').to_dataset()
            assert len(self.losses_ds['T']) == len(ds['T'])
            # Event losses
            ds['event_loss'] = losses['exceed_loss'] * ds['event_prob']
            ds_list.append(ds)
        losses_all = xr.concat(ds_list, dim='index')
        self.losses_ds = xr.merge([self.losses_ds, losses_all])
        self.losses_ds['expected_loss'] = losses_all['event_loss'].sum(dim='T')
        return self

    def insurance_premium(self):
        """economic capital is the exceedance loss for a given return period.
        """
        ec = self.losses_ds['exceed_loss'].sel(T=self.insur_rt_ec)
        self.losses_ds['economic_capital'] = ec
        premium = (self.losses_ds['expected_loss'] + ec*self.insur_rc) * (1+self.insur_margin)
        self.losses_ds['insurance_premium'] = premium
        return self

    def run(self):
        self.estimate_losses().expected_losses().insurance_premium()
        return self

    def summary_to_csv(self):
        self.losses_summary.to_csv(self.output_summary_path, float_format='%.2f')
        return self

    def to_gpkg(self):
        """Write to the disk a gpkg map of combined financials.
        """
        ds = self.losses_ds[['exposure_class', 'aggregation_name', 'size',
                            'geometry', 'building_value', 'expected_loss',
                            'economic_capital', 'insurance_premium']]
        # print(ds)
        df = ds.to_dataframe()
        gdf = gpd.GeoDataFrame(df, geometry='geometry')
        # print(gdf.head())
        gdf.to_file(self.output_summary_map_path, driver='GPKG')
        return self


def main():
    financial_model = FinancialComputation(CONF_FILE)
    financial_model.run()
    financial_model.to_gpkg()
    # print(financial_model.losses_summary.head())
    financial_model.summary_to_csv()
    return None


if __name__ == "__main__":
    main()
