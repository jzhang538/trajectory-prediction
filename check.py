#This is the file to check the accuracy of preprocessed data before data filtering

import scipy.io as scp

mat_file='./TrainSet.mat'
D = scp.loadmat(mat_file)['traj']
T = scp.loadmat(mat_file)['tracks']
# self.T = scp.loadmat(mat_file)['tracks']

ac=0
m=-1
v_id=-1
f_id=-1
ds_id=-1

def check_neighbor_ids(dataset_id,vehicle_id,frame_id):
    nbrs_list=[]
    ref_x=0
    ref_y=0
    for i in range(len(D)):
        if(D[i][0]==dataset_id and D[i][1]==vehicle_id and D[i][2]==frame_id):
            for j in range(int(D[i][10])):
                nbr = D[i][10+1+j]
                print(nbr)
                nbrs_list.append([dataset_id,nbr,frame_id])
            ref_x=D[i][3]
            ref_y=D[i][4]
            ref_l=D[i][5]
            break
    print("Ref Position:",ref_x,ref_y,ref_l)

    for i in range(len(nbrs_list)):
        nbr=nbrs_list[i]
        print(nbr)

        nbr_track=T[int(nbr[0])-1][int(nbr[1])-1].transpose()
        #print(nbr_track)
        #break
        for j in range(len(nbr_track)):
            if(int(nbr_track[j][0])==nbr[2]):
                nx=nbr_track[j][1]
                ny=nbr_track[j][2]
                dx=nx-ref_x
                dy=ny-ref_y
                dis=dx*dx+dy*dy

                print("Neighbor Position {}:".format(i),nx,ny,dx,dy,dis)
                break



for i in range(len(D)):
    num_neighbors=D[i][10]
    print(num_neighbors)
    ac+=num_neighbors

    if(num_neighbors>m):
        v_id=D[i][1]
        ds_id=D[i][0]
        f_id=D[i][2]
        m=num_neighbors

print("Max num:",m)
print('Average num',ac/len(D))
print("Vehicle Info:",ds_id,v_id,f_id)
check_neighbor_ids(int(ds_id),int(v_id),int(f_id))