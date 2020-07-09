# This is the data preprocessing file to guarantee that neighboring vehicles' trajectories are in the prediction range
import numpy as np
import scipy.io as scp

mat_file='./data/TestSet.mat'
save_path='./data/Test.mat'
# mat_file='./data/TestSet.mat'
# save_path='./data/Test.mat'
D = scp.loadmat(mat_file)['traj']
T = scp.loadmat(mat_file)['tracks']

filter_D=True
add_Attr_T =False

def check_neighbor(vehId,t,dsId):
    vehTrack = T[dsId-1][vehId-1].transpose()
    idx = np.argwhere(vehTrack[:, 0] == t).item()
    l = len(vehTrack) - (idx+1)

    # Length of History Trajectory should >=32
    flag=True
    if(idx<32):
        flag=False
    # Length of Future Trajectory should >=52
    if(l<52):
        flag=False
    return flag

def cal_neighbor_dis(vehId,t,dsId,j,refx,refy):
    vehTrack = T[dsId-1][vehId-1].transpose()    
    vehPos = vehTrack[np.where(vehTrack[:,0]==t)][0,1:3]

    dis=(refx-vehPos[0])*(refx-vehPos[0])+(refy-vehPos[1])*(refy-vehPos[1])
    return([j,dis])

if filter_D:
    th_low=4 #Threshold number of neighboring vehicles
    th_high=16
    cnt=0 # Number of available cases
    cnt_f=0 # Number of failure cases
    cnt_dup=0
    drop_list=[]
    higher_list=[]
    lower_list=[]
    ok=0
    num_idx = 64-1
    grid_st_idx = 13-1

    print("Length of D before filtering:", len(D))
    for i in range(len(D)):
        nbrs_num=int(D[i][num_idx])
        ds_id=int(D[i][0])
        v_id=int(D[i][1])
        f_id=int(D[i][2])
        lx=D[i][3]
        ly=D[i][4]

        # Grid Duplicate
        grid_ct=0
        for j in range(51):
            nbr=int(D[i][grid_st_idx+j])
            if(nbr!=0):
                grid_ct+=1
        if(grid_ct!=nbrs_num):
            drop_list.append(i)
            cnt_dup+=1
            continue


        # If the number of neighboring vehicle is smaller than the threshold: 
        if(nbrs_num<th_low):
            lower_list.append([ds_id,v_id,f_id])
            drop_list.append(i)
            continue

        # If the number of neighboring vehicle is larger than the threshold, keep 'th_high' number of vehicles
        if(nbrs_num>th_high):
            higher_list.append([ds_id,v_id,f_id])
            dis_list=[]

            for j in range(51):
                nbr=int(D[i][grid_st_idx+j])
                if(nbr!=0):
                    dis_list.append(cal_neighbor_dis(nbr,f_id,ds_id,j,lx,ly))
            dis_list=sorted(dis_list,key=(lambda x:x[1]))

            for j in range(th_high,nbrs_num):
                D[i][grid_st_idx+dis_list[j][0]]=0
            D[i][num_idx] = th_high
        
        # Check neighboring vehicles:    
        flag=True
        for j in range(51):
            nbr=int(D[i][grid_st_idx+j])
            if(nbr!=0):
                flag=check_neighbor(nbr,f_id,ds_id)
                if(flag==False):
                    break   
        if(flag==True):
            cnt+=1
        else:
            drop_list.append(i)
            cnt_f+=1

        if(i%10000==0):
            print(i)

    D=np.delete(D,drop_list,axis=0)
    print("Lower than threshold",len(lower_list),"Higher than threshold",len(higher_list),"Failure drop",cnt_f,"Duplicate drop",cnt_dup,"Total drop",len(drop_list))
    print("Length of D after filtering", len(D), cnt)

# if add_Attr_T:
#     print((T[0][:]))
#     print(T.shape)
    # for i in range(len(T)):
        # ds_id=int(D[i][0])
        # v_id=int(D[i][1])
        # f_id=int(D[i][2])
        # lx=T[i][3]
        # ly=T[i][4]

scp.savemat(save_path,{'traj':D, 'tracks':T})

