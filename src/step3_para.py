import os
os.environ['HOME'] ='/home/ohno'
import numpy as np
import pandas as pd
import time
import sys
from tqdm import tqdm
import argparse
import numpy as np
from scipy import signal
import scipy.spatial.distance as distance
import random

INTERACTION_PATH = os.path.join(os.environ['HOME'],'Working/interlayer_interaction/')
sys.path.append(INTERACTION_PATH)

from make_8_para import exec_gjf
from step3_para_vdw import get_c_vec_vdw
from step3_para_vdw import detect_peaks
from utils import get_E

def init_process(args):
    # 数理モデル的に自然な定義の元のparams initリスト: not yet
    # 結晶学的に自然なパラメータへ変換: not yet
    auto_dir = args.auto_dir
    
    monomer_name = args.monomer_name
    
    os.makedirs(auto_dir, exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussian'), exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussview'), exist_ok=True)
    ### TODO
    def get_init_para_csv(auto_dir,monomer_name,params_dict):
        a_ = params_dict['a']; b_ = params_dict['b']
        R3 = params_dict['R3']; R4 = params_dict['R4']; theta = params_dict['theta']
        init_params_csv = os.path.join(auto_dir, 'step3_para_init_params.csv')
        init_para_list=[]
        z_2dlist=get_c_vec_vdw(monomer_name,R3,R4,a_,b_,theta)##z_2dlistでは後ろでRcの極小を得るために正負反転したものを出力
        Rb_list=[np.round(Rb,1) for Rb in np.linspace(-np.round(b_/2,1),np.round(b_/2,1),int(np.round(2*np.round(b_/2,1)/0.1))+1)]
        Ra_list=[np.round(Ra,1) for Ra in np.linspace(-np.round(a_/2,1),np.round(a_/2,1),int(np.round(2*np.round(a_/2,1)/0.1))+1)]
        xyz=[]
        mask=detect_peaks(z_2dlist, filter_size=7).mask###このfilterとorderの調整　-Rcの極大を探す
        for i  in range(len(Ra_list)):
            for j in range(len(Rb_list)):
                if str(mask[i][j])=='False':
                    xyz.append([Ra_list[i],Rb_list[j],-z_2dlist[i][j]])##Rcの極小値とRa,Rbを出力
                else:
                    continue
        if len(xyz)>0:
            for i in range(len(xyz)):
                init_para_list.append([a_,b_,theta,R3,R4,xyz[i][0],xyz[i][1],xyz[i][2],'NotYet'])

        df_init_params = pd.DataFrame(np.array(init_para_list),columns = ['a','b','theta','R3','R4','cx','cy','cz','status'])##いじる
        df_init_params.to_csv(init_params_csv,index=False)
    
    params_dict_twist={'a':7.2,'b':6.0,'R3':0,'R4':0,'theta':25}##########ここは入力or読み取り
    get_init_para_csv(auto_dir,monomer_name,params_dict_twist)
    
    auto_csv_path = os.path.join(auto_dir,'step3_para.csv')
    if not os.path.exists(auto_csv_path):        
        df_E = pd.DataFrame(columns = ['a','b','theta','R3','R4','E','E_i0','E_ip1','E_ip2','E_ip3','E_ip4','E_it1','E_it2','E_it3','E_it4','machine_type','status','file_name'])##いじる
    else:
        df_E = pd.read_csv(auto_csv_path)
        df_E = df_E[df_E['status']!='InProgress']
    df_E.to_csv(auto_csv_path,index=False)

    df_init=pd.read_csv(os.path.join(auto_dir,'step3_para_init_params.csv'))
    df_init['status']='NotYet'
    df_init.to_csv(os.path.join(auto_dir,'step3_para_init_params.csv'),index=False)

def main_process(args):
    os.chdir(os.path.join(args.auto_dir,'gaussian'))
    isOver = False
    while not(isOver):
        #check
        isOver = listen(args)
        time.sleep(1)

def listen(args):
    auto_dir = args.auto_dir
    monomer_name = args.monomer_name
    num_nodes = args.num_nodes
    isTest = args.isTest
    isInterlayer =args.isInterlayer
    #### TODO
    fixed_param_keys = ['a','b','theta','R3','R4']
    opt_param_keys = ['cx','cy','cz']

    auto_csv = os.path.join(auto_dir,'step3_para.csv')
    df_E = pd.read_csv(auto_csv)
    df_queue = df_E.loc[df_E['status']=='InProgress',['machine_type','file_name']]
    machine_type_list = df_queue['machine_type'].values.tolist()
    len_queue = len(df_queue)
    maxnum_machine2 = 3#num_nodes/2 if num_nodes%2==0 else (num_nodes+1)/2
    
    for idx,row in zip(df_queue.index,df_queue.values):
        machine_type,file_name = row
        log_filepath = os.path.join(*[auto_dir,'gaussian',file_name])
        if not(os.path.exists(log_filepath)):#logファイルが生成される直前だとまずいので
            continue
        E_list=get_E(log_filepath)
        if len(E_list)!=9:##計算する層状の分子数
            continue
        else:##エネルギーの内訳全般
            len_queue-=1;machine_type_list.remove(machine_type)
            Ei0=float(E_list[0]);Eip1=float(E_list[1]);Eip2=float(E_list[2]);Eip3=float(E_list[3]);Eip4=float(E_list[4]);Eit1=float(E_list[5]);Eit2=float(E_list[6]);Eit3=float(E_list[7]);Eit4=float(E_list[8])##ここも計算する分子数に合わせて調整
            E = Ei0+Eip1+Eip2+Eip3+Eip4+Eit1+Eit2+Eit3+Eit4
            #### TODO
            df_E.loc[idx, ['E_i0','E_it1','E_it2','E_it3','E_it4','E_ip1','E_ip2','E_ip3','E_ip4','E','status']] = [Eit1,Eit2,Eit3,Eit4,Eip1,Eip2,Eip3,Eip4,E,'Done']
            df_E.to_csv(auto_csv,index=False)
            break#2つ同時に計算終わったりしたらまずいので一個で切る
    isAvailable = len_queue < num_nodes 
    machine2IsFull = machine_type_list.count(2) >= maxnum_machine2
    machine_type = 1 if machine2IsFull else 2
    if isAvailable:
        params_dict = get_params_dict(auto_dir,num_nodes, fixed_param_keys, opt_param_keys)
        if len(params_dict)!=0:#終わりがまだ見えないなら
            alreadyCalculated = check_calc_status(auto_dir,params_dict)
            if not(alreadyCalculated):
                #### TODO
                file_name = exec_gjf(auto_dir, monomer_name,params_dict, machine_type,isInterlayer=isInterlayer,isTest=isTest)##paramsdictとか
                df_newline = pd.Series({**params_dict,'E':0.,'E_p':0.,'E_t':0.,'machine_type':machine_type,'status':'InProgress','file_name':file_name})
                df_E=df_E.append(df_newline,ignore_index=True)
                df_E.to_csv(auto_csv,index=False)
    
    init_params_csv=os.path.join(auto_dir, 'step3_para_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_init_params_done = filter_df(df_init_params,{'status':'Done'})
    isOver = True if len(df_init_params_done)==len(df_init_params) else False
    return isOver

def check_calc_status(auto_dir,params_dict):
    df_E= pd.read_csv(os.path.join(auto_dir,'step3_para.csv'))
    if len(df_E)==0:
        return False
    df_E_filtered = filter_df(df_E, params_dict)
    df_E_filtered = df_E_filtered.reset_index(drop=True)
    try:
        status = get_values_from_df(df_E_filtered,0,'status')
        return status=='Done'
    except KeyError:
        return False

def get_params_dict(auto_dir, num_nodes, fixed_param_keys, opt_param_keys):
    """
    前提:
        step3_para_init_params.csvとstep3_para.csvがauto_dirの下にある
    """
    init_params_csv=os.path.join(auto_dir, 'step3_para_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_cur = pd.read_csv(os.path.join(auto_dir, 'step3_para.csv'))
    df_init_params_inprogress = df_init_params[df_init_params['status']=='InProgress']
    #fixed_param_keys = ['a','b','theta','R3','R4']     opt_param_keys = ['cx','cy','cz']


    #最初の立ち上がり時
    if len(df_init_params_inprogress) < num_nodes:
        df_init_params_notyet = df_init_params[df_init_params['status']=='NotYet']
        for index in df_init_params_notyet.index:
            df_init_params = update_value_in_df(df_init_params,index,'status','InProgress')
            df_init_params.to_csv(init_params_csv,index=False)
            params_dict = df_init_params.loc[index,fixed_param_keys+opt_param_keys].to_dict()
            return params_dict
    for index in df_init_params.index:
        df_init_params = pd.read_csv(init_params_csv)
        init_params_dict = df_init_params.loc[index,fixed_param_keys+opt_param_keys].to_dict()
        fixed_params_dict = df_init_params.loc[index,fixed_param_keys].to_dict()
        ### TODO
        isDone, opt_params_dict = get_opt_params_dict(df_cur, init_params_dict,fixed_params_dict)##TF cx cy czを出力
        if isDone:
            # df_init_paramsのstatusをupdate
            df_init_params = update_value_in_df(df_init_params,index,'status','Done')
            if np.max(df_init_params.index) < index+1:
                status = 'Done'
            else:
                status = get_values_from_df(df_init_params,index+1,'status')
            df_init_params.to_csv(init_params_csv,index=False)
            
            if status=='NotYet':                
                opt_params_dict = get_values_from_df(df_init_params,index+1,opt_param_keys)
                df_init_params = update_value_in_df(df_init_params,index+1,'status','InProgress')
                df_init_params.to_csv(init_params_csv,index=False)
                return {**fixed_params_dict,**opt_params_dict}
            else:
                continue

        else:
            df_inprogress = filter_df(df_cur, {**fixed_params_dict,**opt_params_dict,'status':'InProgress'})
            if len(df_inprogress)>=1:
                continue
            return {**fixed_params_dict,**opt_params_dict}
    return {}
        
def get_opt_params_dict(df_cur, init_params_dict,fixed_params_dict):
    df_val = filter_df(df_cur, fixed_params_dict)
    cx_init_prev = init_params_dict['cx']; cy_init_prev = init_params_dict['cy']; cz_init_prev = init_params_dict['cz']
    
    while True:
        E_list=[];cxyz_list=[]
        for cx in [cx_init_prev-0.1,cx_init_prev,cx_init_prev+0.1]:
            for cy in [cy_init_prev-0.1,cy_init_prev,cy_init_prev+0.1]:
                 for cz in [cz_init_prev-0.1,cz_init_prev,cz_init_prev+0.1]:  
                    cx = np.round(cx,1);cy = np.round(cy,1);cz= np.round(cz,1)
                    df_val_cxyz = df_val[
                    (df_val['cx']==cx)&(df_val['cy']==cy)&(df_val['cz']==cz)
                    (df_val['status']=='Done')
                       ]
                    if len(df_val_cxyz)==0:
                         return False,{'cx':cx,'cy':cy,'cz':cz}
                    cxyz_list.append([cx,cy,cz]);E_list.append(df_val_cxyz['E'].values[0])
        cx_init,cy_init,cz_init = cxyz_list[np.argmin(np.array(E_list))]
        if cx_init==cx_init_prev and cy_init==cy_init_prev and cz_init==cz_init_prev:
            return True,{'cx':cx_init,'cy':cy_init, 'cz':cz_init}
        else:
             cx_init_prev=cx_init;cy_init_prev=cy_init;cz_init_prev=cz_init
    
def get_values_from_df(df,index,key):
    return df.loc[index,key]

def update_value_in_df(df,index,key,value):
    df.loc[index,key]=value
    return df

def filter_df(df, dict_filter):
    query = []
    for k, v in dict_filter.items():
        if type(v)==str:
            query.append('{} == "{}"'.format(k,v))
        else:
            query.append('{} == {}'.format(k,v))
    df_filtered = df.query(' and '.join(query))
    return df_filtered

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--init',action='store_true')
    parser.add_argument('--isTest',action='store_true')
    parser.add_argument('--isInterlayer',action='store_true')##追加
    parser.add_argument('--auto-dir',type=str,help='path to dir which includes gaussian, gaussview and csv')
    parser.add_argument('--monomer-name',type=str,help='monomer name')
    parser.add_argument('--num-nodes',type=int,help='num nodes')
    
    args = parser.parse_args()

    if args.init:
        print("----initial process----")
        init_process(args)
    
    print("----main process----")
    main_process(args)
    print("----finish process----")