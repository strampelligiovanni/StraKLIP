'''
This module implements the ShowRoom class, which allow the user through a visual aid 
to check the final product of the PSF subtraction process using ipywidgets.
'''
# import sys
# sys.path.append('/')
# from pipeline_config import path2data
from straklip.utils.utils_tile import load_image
from straklip.utils.utils_dataframe import update_type
from straklip.utils.ancillary import get_monitor_from_coord
from straklip.utils.utils_false_positives import FP_analysis
from straklip.tiles import Tile
from straklip.steps.klipphotometry import update_median_candidates_tile

import numpy as np
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

import ipywidgets as widgets
import tkinter as tk
import functools
import warnings
from scipy.stats import sigmaclip
warnings.filterwarnings('ignore', message='not allowed')

class ShowRoom():
    def __init__(self,DF,DF_fk=None):
        self.DF=DF
        self.DF_fk=DF_fk
        self.v=None

    def matchingKeys(self, dic, searchString):
        return [[key, val] for key, val in dic.items() if searchString in val]

    def matchingExactKeys(self, dic, searchString):
        return [[key, val] for key, val in dic.items() if searchString == val]

    def selectors(self,unq_ids_list=[],CRKmode=False):
        orig_column_names=[('SCI'),('ERR'),('DQ'),('CRCLEANSCI')]
        KLIP_column_names=[]
        MODEL_column_names=[]
        # if CRKmode: pre='crclean_'
        # else: pre=''

        for Kmode in self.DF.kmodes:
            KLIP_column_names.append((Kmode))
        for Kmode in self.DF.kmodes:
            MODEL_column_names.append((Kmode))

        self.ids_dropdown=widgets.SelectionSlider(
            options=unq_ids_list,
            value=unq_ids_list[0],
            description='AVG IDs:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
        )

        self.ids_progress=widgets.IntProgress(
            description='PRG IDs:',
            min=0,
            max=max(unq_ids_list)
        )
        self.orig_column_dropdown=widgets.Dropdown(
            options=orig_column_names,
            description='TILE:',
            # value='%sdata'%pre
        )
        # self.MODEL_column_dropdown=widgets.Dropdown(
        #     options=MODEL_column_names,
        #     description='TILE:',
        #     # value='Model %i'%(Kmode)
        # )
        self.MODEL_column_dropdown=widgets.SelectionSlider(
            options=MODEL_column_names,
            value=MODEL_column_names[0],
            description='Kmode:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
        )
        # self.KLIP_column_dropdown=widgets.Dropdown(
        #     options=KLIP_column_names,
        #     description='TILE:',
        #     # value='%sKmode%i'%(pre,Kmode)
        # )

        self.KLIP_column_dropdown=widgets.SelectionSlider(
            options=KLIP_column_names,
            value=KLIP_column_names[0],
            description='Kmode:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
        )

        self.cmap_column_dropdown=widgets.Dropdown(
            options=['magma','magma_r','plasma','plasma_r','viridis','viridis_r','Greys','Greys_r'],
            value='magma',
            description='CMAP:',
        )
        self.simplenorm_column_dropdown=widgets.Dropdown(
            options=['log','sqrt','linear', 'power'],
            value='log',
            description='SNORM:',
        )
        self.power_textbox=widgets.FloatText(
            value=1,
            description='POWER:',
            disabled=False
        )
        self.log_textbox=widgets.FloatText(
            value=1000,
            description='LOG:',
            disabled=False
        )
        self.crange_slider=widgets.FloatRangeSlider(
            value=[0,100.],
            min=0,
            max=100.0,
            step=0.1,
            description='CRANGE:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
        )
        self.power_slider=widgets.FloatSlider(
            value=2,
            min=0,
            max=10.0,
            step=0.1,
            description='POWER:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
        )
        self.show_mvs_check=widgets.Checkbox(
            value=True,
            description='show_mvs',
            disabled=False,
            indent=False,
            layout=widgets.Layout(width='auto')
        )
        self.show_FP_check=widgets.Checkbox(
            value=False,
            description='show_FP',
            disabled=False,
            indent=False,
            layout=widgets.Layout(width='auto')
        )
        self.xy_m_check=widgets.Checkbox(
            value=False,
            description='xy_m',
            disabled=False,
            indent=False,
            layout=widgets.Layout(width='auto')
        )
        self.xy_cen_check=widgets.Checkbox(
            value=False,
            description='xy_cen',
            disabled=False,
            indent=False,
            layout=widgets.Layout(width='auto')
        )
        self.legend_check=widgets.Checkbox(
            value=False,
            description='legend',
            disabled=False,
            indent=False,
            layout=widgets.Layout(width='auto')
        )
        self.cbar_check=widgets.Checkbox(
            value=True,
            description='cbar',
            disabled=False,
            indent=False,
            layout=widgets.Layout(width='auto')
        )
        self.good_sample_button=widgets.Button(
            button_style='info', # 'success', 'info', 'warning', 'danger' or ''
            description='good sample',
            layout=widgets.Layout(width='auto')
        )
        self.good_candidate_sample_button=widgets.Button(
            button_style='info', # 'success', 'info', 'warning', 'danger' or ''
            description='good sample',
            layout=widgets.Layout(width='auto')
        )
        self.psf_sample_button=widgets.Button(
            button_style='info', # 'success', 'info', 'warning', 'danger' or ''
            description='psf sample',
            layout=widgets.Layout(width='auto')
        )
        self.unresolved_sample_button=widgets.Button(
            button_style='info', # 'success', 'info', 'warning', 'danger' or ''
            description='unresolved sample',
            layout=widgets.Layout(width='auto')
        )
        self.known_sample_buttone=widgets.Button(
            button_style='info', # 'success', 'info', 'warning', 'danger' or ''
            description='known sample',
            layout=widgets.Layout(width='auto')
        )
        self.bad_sample_button=widgets.Button(
            button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
            description='bad sample',
            layout=widgets.Layout(width='auto')
        )
        self.bad_candidate_sample_button=widgets.Button(
            button_style='warning', # 'success', 'info', 'warning', 'success' or ''
            description='bad sample',
            layout=widgets.Layout(width='auto')
        )
        self.change_flag_filters=widgets.Text(
            placeholder='Choose filters to applay flag [a=all; enter to apply]',
            value=None,
            disabled=False,
            layout=widgets.Layout(height="auto", width="auto")
        )
        self.change_flag_mvs_ids=widgets.Text(
            placeholder='Choose mvs_ids to applay flag [a=all; enter to apply]',
            value=None,
            disabled=False,
            layout=widgets.Layout(height="auto", width="auto")
        )
        self.change_type=widgets.Text(
            placeholder='Choose type to applay to this unq_ids [enter to apply]',
            value=None,
            disabled=False,
            layout=widgets.Layout(height="auto", width="auto")
        )
        self.caption = widgets.Label(value='')

    
    def upon_submitted_filter_flag_text(self,wdgt,rs_=[]):
        global filters_list_selected_global
        id,out1a=rs_
        if wdgt.value=='a' or wdgt.value=='':filters_list_selected_global=self.DF.filters
        else: 
            filters_list_selected_global=[]
            for filter in wdgt.value.split(','):
                if filter not in self.DF.filters:
                    with out1a:
                        print('%s filter is not present in the standard filters list from the header %s. Please check.'%(filter,self.DF.filters))
                    display(out1a)
                    self.display_mvs_flags(self.DF.crossmatch_ids_df.loc[self.DF.crossmatch_ids_df.unq_ids==id].mvs_ids.values,out1a)

                else:
                    filters_list_selected_global.append(filter)
    
    def upon_submitted_mvs_ids_text(self,wdgt,rs_=[]):
        global mvs_ids_list_selected_global

        id,out1a=rs_
        mvs_ids_list=self.DF.crossmatch_ids_df.loc[self.DF.crossmatch_ids_df.unq_ids==id].mvs_ids.values
        if wdgt.value=='a' or wdgt.value=='': mvs_ids_list_selected_global=mvs_ids_list 
        else: 
            mvs_ids_list_selected_global=[]
            for mvs_ids in wdgt.value.split(','):
                if mvs_ids not in mvs_ids_list.astype(str): 
                    with out1a:
                        print('%s mvs_ids is not present in the selected mvs_ids_list from this star %s. Please check.'%(mvs_ids,mvs_ids_list))
                    display(out1a)
                    self.display_mvs_flags(self.DF.crossmatch_ids_df.loc[self.DF.crossmatch_ids_df.unq_ids==id].mvs_ids.values,out1a)

                else:
                    mvs_ids_list_selected_global.append(int(mvs_ids))


    def upon_submitted_type_text(self,wdgt,rs_=[]):
        id,out_target=rs_
        if wdgt.value!='':
            self.DF.unq_targets_df.loc[self.DF.unq_targets_df.unq_ids==id,'type']=int(wdgt.value)
            # self.display_unq_df(id,out_target)

    def upon_clicked_on_target_flag_button(self,wdgt,rs_=[]):
        id,out1a,out_target=rs_
        id_type=self.DF.unq_targets_df.loc[self.DF.unq_targets_df.unq_ids==id].type.values[0]
        if wdgt.description in ['good sample','bad sample']:
            if wdgt.description =='good sample':
                for filter in filters_list_selected_global:
                    mvs_ids=self.DF.mvs_targets_df.loc[self.DF.mvs_targets_df.mvs_ids.isin(mvs_ids_list_selected_global)&~(self.DF.mvs_targets_df[f'flag_{filter}'].str.contains('rejected'))].mvs_ids.unique()
                    self.DF.mvs_targets_df.loc[self.DF.mvs_targets_df.mvs_ids.isin(mvs_ids),[f'flag_{filter}']]='good_target'
                self.DF.unq_targets_df.loc[self.DF.unq_targets_df.unq_ids.isin([id])&self.DF.unq_targets_df['type'].isin([0,1,2,3]),['type']]=1
            elif wdgt.description =='bad sample':
                for filter in filters_list_selected_global:
                    mvs_ids=self.DF.mvs_targets_df.loc[self.DF.mvs_targets_df.mvs_ids.isin(mvs_ids_list_selected_global)].mvs_ids.unique()
                    self.DF.mvs_targets_df.loc[self.DF.mvs_targets_df.mvs_ids.isin(mvs_ids),[f'flag_{filter}']]='rejected'
                    self.DF.make_unq_tiles_and_photometry(filter,unq_ids_test_list=[id],la_cr_remove=self.DF.header_df.loc['la_cr_remove','Values'],parallel_runs=False,Python_origin=self.DF.header_df.loc['Python_origin','Values'],kill_plots=True,verbose=False)
        else:
            if id_type in [0,1,2,3]:
                if wdgt.description =='psf sample':
                    for filter in filters_list_selected_global:
                        mvs_ids=self.DF.mvs_targets_df.loc[self.DF.mvs_targets_df.mvs_ids.isin(mvs_ids_list_selected_global)&~(self.DF.mvs_targets_df[f'flag_{filter}'].str.contains('rejected'))].mvs_ids.unique()
                        self.DF.mvs_targets_df.loc[self.DF.mvs_targets_df.mvs_ids.isin(mvs_ids),[f'flag_{filter}']]='good_psf'
                    self.DF.unq_targets_df.loc[self.DF.unq_targets_df.unq_ids.isin([id]),['type']]=1
                elif wdgt.description =='unresolved sample':
                    for filter in filters_list_selected_global:
                        mvs_ids=self.DF.mvs_targets_df.loc[self.DF.mvs_targets_df.mvs_ids.isin(mvs_ids_list_selected_global)&~(self.DF.mvs_targets_df[f'flag_{filter}'].str.contains('rejected'))].mvs_ids.unique()
                        self.DF.mvs_targets_df.loc[self.DF.mvs_targets_df.mvs_ids.isin(mvs_ids),[f'flag_{filter}']]='unresolved_double'
                    self.DF.unq_targets_df.loc[self.DF.unq_targets_df.unq_ids.isin([id]),['type']]=2
                elif wdgt.description =='known sample':
                    for filter in filters_list_selected_global:
                        mvs_ids=self.DF.mvs_targets_df.loc[self.DF.mvs_targets_df.mvs_ids.isin(mvs_ids_list_selected_global)&~(self.DF.mvs_targets_df[f'flag_{filter}'].str.contains('rejected'))].mvs_ids.unique()
                        self.DF.mvs_targets_df.loc[self.DF.mvs_targets_df.mvs_ids.isin(mvs_ids),[f'flag_{filter}']]='known_sample'
                    self.DF.unq_targets_df.loc[self.DF.unq_targets_df.unq_ids.isin([id]),['type']]=3
            
            else:
                with out1a:
                    print('Type %s source can\'t be changed to %s'%(id_type,wdgt.description))
                display(out1a)

        update_type(self.DF,id)
        # self.display_unq_df(id,out_target)
        self.display_mvs_flags(self.DF.crossmatch_ids_df.loc[self.DF.crossmatch_ids_df.unq_ids==id].mvs_ids.values,out1a)

    def upon_clicked_on_candidate_flag_button(self,wdgt,rs_=[]):
        id,out1c,out_candidate=rs_
        # elif wdgt.description =='bad sample':
        for filter in filters_list_selected_global:
            mvs_ids=self.DF.mvs_candidates_df.loc[self.DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list_selected_global)].mvs_ids.unique()
            self.DF.mvs_candidates_df.loc[self.DF.mvs_candidates_df.mvs_ids.isin(mvs_ids),[f'flag_{filter}']]='rejected'            
            self.DF.make_unq_tiles_and_photometry(filter,unq_ids_test_list=[id],la_cr_remove=self.DF.header_df.loc['la_cr_remove','Values'],parallel_runs=False,Python_origin=self.DF.header_df.loc['Python_origin','Values'],kill_plots=True,verbose=False)
        if CRKmode_global: label='crclean_data'
        else: label='data'
        if np.all(self.DF.mvs_candidates_df.loc[self.DF.mvs_candidates_df.mvs_ids.isin(self.DF.crossmatch_ids_df.loc[self.DF.crossmatch_ids_df.unq_ids==id].mvs_ids),[f'flag_{filter}' for filter in self.DF.filters]]=='rejected'):
            self.DF.mvs_candidates_df=self.DF.mvs_candidates_df.loc[~self.DF.mvs_candidates_df.mvs_ids.isin(self.DF.crossmatch_ids_df.loc[self.DF.crossmatch_ids_df.unq_ids==id].mvs_ids)].reset_index(drop=True)
            self.DF.unq_candidates_df=self.DF.unq_candidates_df.loc[~(self.DF.unq_candidates_df.unq_ids==id)].reset_index(drop=True)
        update_median_candidates_tile(self.DF,unq_ids_list=[id],parallel_runs=False,label=label,kill_plots=True)

        # self.display_unq_df(id,out_candidate,candidate=True)
        self.display_mvs_flags(self.DF.crossmatch_ids_df.loc[self.DF.crossmatch_ids_df.unq_ids==id].mvs_ids.values,out1c,candidate=True)
        # with out1c:
        #     print('%s candidate inputs deleted'%id)
        #     out1c.clear_output(wait=True)
        # display(out1c)
        
    def display_unq_df(self,unq_ids,out,candidate=False):
        with out:
            if candidate:display(self.DF.unq_candidates_df.loc[self.DF.unq_candidates_df.unq_ids==unq_ids])
            else:display(self.DF.unq_targets_df.loc[self.DF.unq_targets_df.unq_ids==unq_ids])
            out.clear_output(wait=True)
        display(out)
        del out
        
    def display_mvs_df(self,mvs_ids_list,out,candidate=False):
        with out:
            if candidate:display(self.DF.mvs_candidates_df.loc[self.DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list)])
            else:display(self.DF.mvs_targets_df.loc[self.DF.mvs_targets_df.mvs_ids.isin(mvs_ids_list)])
            out.clear_output(wait=True)
        display(out)
        del out

    def display_mvs_flags(self,mvs_ids_list,out,candidate=False):
        with out:
            if candidate:
                display(self.DF.mvs_candidates_df.loc[self.DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list),['mvs_ids']+[f'flag_{filter}' for filter in self.DF.filters]+[f'kmode_{filter}' for filter in self.DF.filters]+[f'nsigma_{filter}' for filter in self.DF.filters]+[f'sep_{filter}' for filter in self.DF.filters]])
            else:
                display(self.DF.mvs_targets_df.loc[
                            self.DF.mvs_targets_df.mvs_ids.isin(mvs_ids_list), ['mvs_ids'] + [f'flag_{filter}' for
                                                                                              filter in
                                                                                              self.DF.filters]])
            out.clear_output(wait=True)
        return(out)

    # def display_mvs_kmode(self,mvs_ids_list,out):
    #     with out:
    #         print('Candidate kmode:')
    #         display(self.DF.mvs_candidates_df.loc[self.DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list),['mvs_ids']+[f'kmode_{filter}' for filter in self.DF.filters]])
    #         out.clear_output(wait=True)
    #     return(out)

    def display_text_boxes(self,candidate=False):
        if candidate:
            clear_output(wait=True)
            box1=widgets.HBox([self.bad_candidate_sample_button])
            box2=widgets.VBox([self.change_flag_mvs_ids,self.change_flag_filters,self.change_type])
        
        else:
            clear_output(wait=True)
            box1=widgets.HBox([self.good_sample_button,self.psf_sample_button,self.unresolved_sample_button,self.known_sample_buttone,self.bad_sample_button])
            box2=widgets.VBox([self.change_flag_mvs_ids,self.change_flag_filters,self.change_type])
        return(box1,box2)

    def target_buttons_and_text_area(self,id,out_target,out1a):
        self.change_flag_mvs_ids.on_submit(functools.partial(self.upon_submitted_mvs_ids_text,rs_=[id,out1a]))
        self.change_flag_filters.on_submit(functools.partial(self.upon_submitted_filter_flag_text,rs_=[id,out1a]))


        self.good_sample_button.on_click(functools.partial(self.upon_clicked_on_target_flag_button,rs_=[id,out1a,out_target]))
        self.psf_sample_button.on_click(functools.partial(self.upon_clicked_on_target_flag_button,rs_=[id,out1a,out_target]))
        self.unresolved_sample_button.on_click(functools.partial(self.upon_clicked_on_target_flag_button,rs_=[id,out1a,out_target]))
        self.known_sample_buttone.on_click(functools.partial(self.upon_clicked_on_target_flag_button,rs_=[id,out1a,out_target]))
        self.bad_sample_button.on_click(functools.partial(self.upon_clicked_on_target_flag_button,rs_=[id,out1a,out_target]))

    def candidate_buttons_and_text_area(self,id,out_candidate,out1c):
        self.change_flag_mvs_ids.on_submit(functools.partial(self.upon_submitted_mvs_ids_text,rs_=[id,out1c]))
        self.change_flag_filters.on_submit(functools.partial(self.upon_submitted_filter_flag_text,rs_=[id,out1c]))

        # self.good_candidate_sample_button.on_click(functools.partial(self.upon_clicked_on_candidate_flag_button,rs_=[id,out1c,out_candidate]))
        self.bad_candidate_sample_button.on_click(functools.partial(self.upon_clicked_on_candidate_flag_button,rs_=[id,out1c,out_candidate]))


    def build_targets_box(self,id):
        out1a = widgets.Output()
        self.display_text_boxes()
        self.target_buttons_and_text_area(id,out_target_global,out1a)
        box1,box2=self.display_text_boxes()
        box3=self.display_mvs_flags(self.DF.crossmatch_ids_df.loc[self.DF.crossmatch_ids_df.unq_ids==id].mvs_ids.values,out1a)
        display(widgets.HBox([widgets.VBox([box1,box2]),box3]))

    def build_candidates_box(self,id):
        out1c = widgets.Output()
        self.display_text_boxes(candidate=True)
        self.candidate_buttons_and_text_area(id,out_candidate_global,out1c)
        box1,box2=self.display_text_boxes(candidate=True)
        box3=self.display_mvs_flags(self.DF.crossmatch_ids_df.loc[self.DF.crossmatch_ids_df.unq_ids==id].mvs_ids.values,out1c,candidate=True)
        # box4=self.display_mvs_kmode(self.DF.crossmatch_ids_df.loc[self.DF.crossmatch_ids_df.unq_ids==id].mvs_ids.values,out1c)
        display(widgets.HBox([widgets.VBox([box1,box2]),box3]))

    def load_and_plot_tiles(self, mvs_ids_list, nrows, ncols, fig1, ax1, fig2, ax2,unq_column_name,mvs_column_name,
                            id=0, cmap='', simplenorm='',
                            percent=[0, 100], power=1, log=1000, show_mvs=False, xy_m=True, xy_cen=False,
                            legend=True, cbar=True, SNR=False):
        elnoy=0
        for filter in self.DF.filters:
            elnox=0
            if show_mvs:
                for mvs_ids in mvs_ids_list:
                    try:
                        if elnox==nrows-1 and legend: legend_in=True
                        else: legend_in=False
                        ax_in=ax1[elnoy][elnox]
                        ax_in.grid(False)

                        DATA=Tile()
                        Datacube=DATA.load_tile(f'{self.DF.path2out}/mvs_tiles/{filter}/tile_ID{mvs_ids}.fits',
                                       raise_errors=False,return_Datacube=True)
                        mvs_label_dict={Datacube[i].name:i for i in range(len(Datacube))}

                        image=Datacube[mvs_label_dict[mvs_column_name]].data
                        if SNR:
                            filtered_data, lower_bound, upper_bound = sigmaclip(image, low=5, high=5)
                            STD = np.nanstd(filtered_data)
                            image/=STD
                        PA_V3=self.DF.mvs_targets_df.loc[self.DF.mvs_targets_df.mvs_ids==mvs_ids,f'pav3_{filter}'].values[0]
                        ROTA=self.DF.mvs_targets_df.loc[self.DF.mvs_targets_df.mvs_ids==mvs_ids,f'rota_{filter}'].values[0]
                        if not np.isnan(image).all():
                            load_image(image,filter,fig=fig1,ax=ax_in,cmap=cmap,title='%s ID %i PAV3 %.2f ROTA %.2f'%(filter,mvs_ids,PA_V3,ROTA),tile_base=self.DF.tilebase,simplenorm=simplenorm,min_percent=percent[0],max_percent=percent[1],power=power,log=log,xy_m=xy_m,xy_cen=xy_cen,legend=legend_in,cbar=cbar,showplot=False)
                            try:
                                x=self.DF.mvs_candidates_df.loc[self.DF.mvs_candidates_df.mvs_ids==mvs_ids,f'x_tile_{filter}'].values[0]
                                y=self.DF.mvs_candidates_df.loc[self.DF.mvs_candidates_df.mvs_ids==mvs_ids,f'y_tile_{filter}'].values[0]
                                ax_in.plot(x,y,'ok',ms=3)
                            except:pass
                        else:
                            ax_in.axis('off')
                    except:
                        ax_in.axis('off')
                    elnox+=1

            if elnoy==ncols-1 and legend: legend_in=True
            else: legend_in=False
            if fig2!=None:
                ax_in=ax2[elnoy]
                ax_in.grid(False)
                try:
                    DATA=Tile()
                    Datacube=DATA.load_tile(f'{self.DF.path2out}/median_tiles/{filter}/tile_ID{id}.fits',
                                    raise_errors=False,return_Datacube=True)

                    unq_label_dict={Datacube[i].name:i for i in range(len(Datacube))}
                    image=Datacube[unq_label_dict[unq_column_name]].data
                    if SNR:
                        filtered_data, lower_bound, upper_bound = sigmaclip(image, low=5, high=5)
                        STD = np.nanstd(filtered_data)
                        image /= STD
                    load_image(image,filter,fig=fig2,ax=ax_in,cmap=cmap,title='%s'%(filter),tile_base=self.DF.tilebase,
                               simplenorm=simplenorm,min_percent=percent[0],max_percent=percent[1],power=power,log=log,
                               xy_m=xy_m,xy_cen=xy_cen,legend=legend_in,cbar=cbar,showplot=False)
                    try:
                        x=np.nanmedian(self.DF.mvs_candidates_df.loc[self.DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list),f'x_rot_{filter}'].values.astype(float))
                        y=np.nanmedian(self.DF.mvs_candidates_df.loc[self.DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list),f'y_rot_{filter}'].values.astype(float))
                        ax_in.plot(x,y,'ok',ms=3)
                    except:
                       pass
                    else:
                        ax_in.axis('off')
                except:
                    ax_in.axis('off')
            elnoy+=1
        plt.show()
        plt.close('LL')

    def show_target_tiles_and_df(self, id=0, cmap='',unq_column_name='data',mvs_column_name='_Kmode',
                                 simplenorm='', percent=[0, 100], power=1, log=1000, show_mvs=False, xy_m=True,
                                 xy_cen=False, legend=True, cbar=True):

        global mvs_ids_list_selected_global,filters_list_selected_global,out_target_global
        mvs_ids_list_selected_global=self.DF.crossmatch_ids_df.loc[self.DF.crossmatch_ids_df.unq_ids==id].mvs_ids.values
        filters_list_selected_global=self.DF.filters
        
        if unq_column_name in ['edata','dqdata']:unq_column_name='data'

        nrows=len(mvs_ids_list_selected_global)
        ncols=len(self.DF.filters)
        if not show_mvs:

            fig2,ax2=plt.subplots(1,ncols,figsize=(7*ncols+ncols,7))
            if ncols == 1: ax2 = [ax2]

            box_layout = widgets.Layout(display='flex',
                                        # flex_flow='row wrap',
                                        width=width_global)
            out_candidate_global = widgets.Output(layout=box_layout)
            fig1, ax1 = [None, None]
        else:
            fig1,ax1=plt.subplots(ncols,nrows,figsize=(5*nrows+nrows,5*ncols),squeeze=False)
            fig2, ax2 = [None, None]

        box_layout = widgets.Layout(display='flex',
                                    # flex_flow='row wrap',
                                    width=width_global)

        out_target_global = widgets.Output(layout=box_layout)
        self.change_type.on_submit(functools.partial(self.upon_submitted_type_text,rs_=[id,out_target_global]))

        self.load_and_plot_tiles(mvs_ids_list_selected_global,nrows,ncols,fig1,ax1,fig2,ax2,unq_column_name,mvs_column_name,id=id,cmap=cmap,simplenorm=simplenorm,percent=percent,power=power,log=log,show_mvs=show_mvs,xy_m=xy_m,xy_cen=xy_cen,legend=legend,cbar=cbar)

    def show_model_tiles_and_df(self, id=0, cmap='',unq_column_name='data',mvs_column_name='_Kmode',
                                simplenorm='', percent=[0, 100], power=1, log=1000, show_mvs=False, show_FP=False,
                                xy_m=True, xy_cen=False, legend=True, cbar=True):

        global mvs_ids_list_selected_global,filters_list_selected_global,out_model_global
        mvs_column_name=f'MODEL{mvs_column_name}'

        mvs_ids_list_selected_global=self.DF.crossmatch_ids_df.loc[self.DF.crossmatch_ids_df.unq_ids==id].mvs_ids.values
        filters_list_selected_global=self.DF.filters

        nrows=len(mvs_ids_list_selected_global)
        ncols=len(self.DF.filters)
        if not show_mvs:
            fig2,ax2=plt.subplots(1,ncols,figsize=(7*ncols+ncols,7))
            if ncols == 1: ax2 = [ax2]
            fig1, ax1 = [None, None]
        else:
            fig1,ax1=plt.subplots(ncols,nrows,figsize=(5*nrows+nrows,5*ncols),squeeze=False)
            fig2, ax2 = [None, None]
        box_layout = widgets.Layout(display='flex',
                                    # flex_flow='row wrap',
                                    width=width_global)
        out_candidate_global = widgets.Output(layout=box_layout)
        plt.close(fig2)
        fig2,ax2=[None,None]
        self.load_and_plot_tiles(mvs_ids_list_selected_global,nrows,ncols,fig1,ax1,fig2,ax2,unq_column_name,mvs_column_name,id=id,cmap=cmap,simplenorm=simplenorm,percent=percent,power=power,log=log,show_mvs=show_mvs,xy_m=xy_m,xy_cen=xy_cen,legend=legend,cbar=cbar)


    def show_candidate_tiles_and_df(self, id=0, cmap='',unq_column_name='data',mvs_column_name='_Kmode',
                                    simplenorm='', percent=[0, 100], power=1, log=1000, show_mvs=False,
                                    show_FP=False, xy_m=True, xy_cen=False, legend=True, cbar=True):

        global out_candidate_global,mvs_ids_list_selected_global,filters_list_selected_global,out_candidate_global
        if CRKmode_global:
            unq_column_name=f'CRCLEAN_KMODE{unq_column_name}'
        else:
            unq_column_name=f'KMODE{unq_column_name}'

        mvs_ids_list_selected_global=self.DF.crossmatch_ids_df.loc[self.DF.crossmatch_ids_df.unq_ids==id].mvs_ids.values
        filters_list_selected_global=self.DF.filters
        nrows = len(mvs_ids_list_selected_global)
        ncols = len(self.DF.filters)
        if not show_mvs:
            fig2,ax2=plt.subplots(1,ncols,figsize=(7*ncols+ncols,7))
            if ncols == 1: ax2 = [ax2]

            box_layout = widgets.Layout(display='flex',
                                        # flex_flow='row wrap',
                                        width=width_global)
            out_candidate_global = widgets.Output(layout=box_layout)
            fig1, ax1 = [None, None]
        else:
            fig1,ax1=plt.subplots(ncols,nrows,figsize=(5*nrows+nrows,5*ncols),squeeze=False)
            fig2, ax2 = [None, None]

        if CRKmode_global:
            mvs_column_name = f'CRCLEAN_KMODE{mvs_column_name}'
        else:
            mvs_column_name = f'KMODE{mvs_column_name}'

        self.load_and_plot_tiles(mvs_ids_list_selected_global,nrows,ncols,fig1,ax1,fig2,ax2,unq_column_name,mvs_column_name,id=id,cmap=cmap,simplenorm=simplenorm,percent=percent,power=power,log=log,show_mvs=show_mvs,xy_m=xy_m,xy_cen=xy_cen,legend=legend,cbar=cbar)

        if id in self.DF.unq_candidates_df.unq_ids.unique() and show_FP and not self.DF.fk_completeness_df.empty:
            for filter in self.DF.filters:
                FP_analysis(self.DF,id,filter,0.5,showplot=True,nbins=30)

    def show_candidate_SNR_tiles_and_df(self, id=0, cmap='',unq_column_name='data',mvs_column_name='_Kmode',
                                    simplenorm='', percent=[0, 100], power=1, log=1000, show_mvs=False,
                                    show_FP=False, xy_m=True, xy_cen=False, legend=True, cbar=True):

        global out_candidate_global,mvs_ids_list_selected_global,filters_list_selected_global,out_candidate_global
        if CRKmode_global:
            unq_column_name=f'CRCLEAN_KMODE{unq_column_name}'
        else:
            unq_column_name=f'KMODE{unq_column_name}'

        mvs_ids_list_selected_global=self.DF.crossmatch_ids_df.loc[self.DF.crossmatch_ids_df.unq_ids==id].mvs_ids.values
        filters_list_selected_global=self.DF.filters
        nrows = len(mvs_ids_list_selected_global)
        ncols = len(self.DF.filters)
        if not show_mvs:
            fig2,ax2=plt.subplots(1,ncols,figsize=(7*ncols+ncols,7))
            if ncols == 1: ax2 = [ax2]
            fig1, ax1 = [None, None]
        else:
            fig1,ax1=plt.subplots(ncols,nrows,figsize=(5*nrows+nrows,5*ncols),squeeze=False)
            fig2, ax2 = [None, None]

        box_layout = widgets.Layout(display='flex',
                                        # flex_flow='row wrap',
                                        width=width_global)
        out_candidate_global = widgets.Output(layout=box_layout)

        if CRKmode_global:
            mvs_column_name = f'CRCLEAN_KMODE{mvs_column_name}'
        else:
            mvs_column_name = f'KMODE{mvs_column_name}'

        self.load_and_plot_tiles(mvs_ids_list_selected_global,nrows,ncols,fig1,ax1,fig2,ax2,unq_column_name,mvs_column_name,id=id,cmap=cmap,simplenorm=simplenorm,percent=percent,power=power,log=log,show_mvs=show_mvs,xy_m=xy_m,xy_cen=xy_cen,legend=legend,cbar=cbar, SNR=True)

        if id in self.DF.unq_candidates_df.unq_ids.unique() and show_FP and not self.DF.fk_completeness_df.empty:
            for filter in self.DF.filters:
                FP_analysis(self.DF,id,filter,0.5,showplot=True,nbins=30)


    def showroom(self,unq_ids_list=[],type=None,CRKmode=False,companion=False,flags=None,widht_range=1,w=None,h=None):
        global CRKmode_global,width_global#,unq_label_dict_global,mvs_label_dict_global,klip_unq_label_dict_global,klip_mvs_label_dict_global,model_mvs_label_dict_global
        CRKmode_global=CRKmode
        self.new_id=self.DF.crossmatch_ids_df.unq_ids.unique()[0]
        root = tk.Tk()
        current_screen = get_monitor_from_coord(root.winfo_x(), root.winfo_y())
        if w is not None and h is not None:
            print(f'selected screen w,h: {w,h}')
        else:
            w,h=[current_screen.width, current_screen.height]
            print(f'current screen w,h: {w,h}')
        width_global='%spx'%(float(w))
        height='%spx'%(float(h))
        if len(unq_ids_list)==0: 
            if companion:
                unq_ids_list=self.DF.crossmatch_ids_df.loc[self.DF.crossmatch_ids_df.mvs_ids.isin(self.DF.mvs_candidates_df.mvs_ids.unique())].unq_ids.unique()
            elif type!=None:# and flag == None:
                unq_ids_list=self.DF.unq_targets_df.loc[self.DF.unq_targets_df.type==type].unq_ids.unique()
            elif flags!=None:# and flag == None:
                mvs_ids_list=self.DF.mvs_targets_df.loc[self.DF.mvs_targets_df[[f'flag_{filter}' for filter in self.DF.filters]].apply(lambda x: x.str.contains(flags,case=False)).any(axis=1)].mvs_ids.unique()
                unq_ids_list=self.DF.crossmatch_ids_df.loc[self.DF.crossmatch_ids_df.mvs_ids.isin(mvs_ids_list)].unq_ids.unique()
            else: unq_ids_list=self.DF.crossmatch_ids_df.unq_ids.unique()
        unq_ids_list=np.sort(unq_ids_list)
        self.selectors(unq_ids_list=unq_ids_list,CRKmode=CRKmode_global)


        widgets.link((self.ids_dropdown, 'value'), (self.ids_progress, 'value'))
        Target = widgets.interactive_output(self.show_target_tiles_and_df,{'id':self.ids_dropdown,'unq_column_name':self.orig_column_dropdown,'mvs_column_name':self.orig_column_dropdown,'cmap':self.cmap_column_dropdown,'simplenorm':self.simplenorm_column_dropdown,'power':self.power_textbox,'log':self.log_textbox,'percent':self.crange_slider,'show_mvs':self.show_mvs_check,'xy_m':self.xy_m_check,'xy_cen':self.xy_cen_check,'legend':self.legend_check,'cbar':self.cbar_check})
        TFlags = widgets.interactive_output(self.build_targets_box,{'id':self.ids_dropdown})

        if hasattr(self.DF,'unq_candidates_df'):
            Model = widgets.interactive_output(self.show_model_tiles_and_df,{'id':self.ids_dropdown,'unq_column_name':self.orig_column_dropdown,'mvs_column_name':self.MODEL_column_dropdown,'cmap':self.cmap_column_dropdown,'simplenorm':self.simplenorm_column_dropdown,'power':self.power_textbox,'log':self.log_textbox,'percent':self.crange_slider,'show_mvs':self.show_mvs_check,'show_FP':self.show_FP_check,'xy_m':self.xy_m_check,'xy_cen':self.xy_cen_check,'legend':self.legend_check,'cbar':self.cbar_check})
            Candidate = widgets.interactive_output(self.show_candidate_tiles_and_df,{'id':self.ids_dropdown,'unq_column_name':self.KLIP_column_dropdown,'mvs_column_name':self.KLIP_column_dropdown,'cmap':self.cmap_column_dropdown,'simplenorm':self.simplenorm_column_dropdown,'power':self.power_textbox,'log':self.log_textbox,'percent':self.crange_slider,'show_mvs':self.show_mvs_check,'show_FP':self.show_FP_check,'xy_m':self.xy_m_check,'xy_cen':self.xy_cen_check,'legend':self.legend_check,'cbar':self.cbar_check})
            Candidate_SNR = widgets.interactive_output(self.show_candidate_SNR_tiles_and_df,{'id':self.ids_dropdown,'unq_column_name':self.KLIP_column_dropdown,'mvs_column_name':self.KLIP_column_dropdown,'cmap':self.cmap_column_dropdown,'simplenorm':self.simplenorm_column_dropdown,'power':self.power_textbox,'log':self.log_textbox,'percent':self.crange_slider,'show_mvs':self.show_mvs_check,'show_FP':self.show_FP_check,'xy_m':self.xy_m_check,'xy_cen':self.xy_cen_check,'legend':self.legend_check,'cbar':self.cbar_check})
            CFlags = widgets.interactive_output(self.build_candidates_box,{'id':self.ids_dropdown})
        else:
            CFlags= widgets.Output()

        box_layout = widgets.Layout(display='flex',
                                    # flex_flow='row wrap',
                                    width=width_global)
        box_layout2 = widgets.Layout(display='flex',
                                    # flex_flow='row wrap',
                                    width=width_global,
                                    height=height)
                
        target_box=widgets.VBox([self.ids_progress,self.ids_dropdown,self.orig_column_dropdown])
        model_box=widgets.VBox([self.ids_progress,self.ids_dropdown,self.MODEL_column_dropdown])
        companion_box=widgets.VBox([self.ids_progress,self.ids_dropdown,self.KLIP_column_dropdown])
        companion_SNR_box=widgets.VBox([self.ids_progress,self.ids_dropdown,self.KLIP_column_dropdown])
        box1=widgets.HBox([self.show_mvs_check,self.show_FP_check,self.xy_m_check,self.xy_cen_check,self.legend_check,self.cbar_check],layout=box_layout)
        box2=widgets.VBox([self.cmap_column_dropdown,self.simplenorm_column_dropdown,self.power_textbox,self.log_textbox,self.crange_slider])
        
        target_hbox=widgets.HBox([target_box,box2,TFlags],layout=box_layout2)
        model_hbox=widgets.HBox([model_box,box2,CFlags],layout=box_layout2)
        companion_hbox=widgets.HBox([companion_box,box2,CFlags],layout=box_layout2)
        companion_SNR_hbox=widgets.HBox([companion_SNR_box,box2,CFlags],layout=box_layout2)

        #create tabs
        tab_nest = widgets.Tab(children=[target_hbox, model_hbox, companion_hbox, companion_SNR_hbox])
        tab_nest.set_title(0, 'Target')
        tab_nest.set_title(1, 'Model')
        tab_nest.set_title(2, 'Candidate')
        tab_nest.set_title(3, 'Candidate SNR')

        VBox1=widgets.VBox([box1,target_hbox,Target],layout=box_layout)
        if hasattr(self.DF,'unq_candidates_df'):
            VBox2=widgets.VBox([box1,model_hbox,Model],layout=box_layout)
            VBox3=widgets.VBox([box1,companion_hbox,Candidate],layout=box_layout)
            VBox4=widgets.VBox([box1,companion_SNR_hbox,Candidate_SNR],layout=box_layout)
            tab_nest.children=[VBox1,VBox2,VBox3,VBox4]
        else:
            tab_nest.children=[VBox1]
        display(tab_nest)

