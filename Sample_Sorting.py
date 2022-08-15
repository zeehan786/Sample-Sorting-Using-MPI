from numpy import random
from mpi4py import MPI
import numpy as np
import math

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
size = comm_world.Get_size()
status = MPI.Status()


start = MPI.Wtime()

def MergeLow(mykeys, recvkeys, sample_size):
    temp_keys = np.zeros((sample_size), dtype=np.intc)
    m_i = 0
    r_i = 0
    t_i = 0

    while t_i < sample_size:
        if mykeys[m_i] <= recvkeys[r_i]:
            temp_keys[t_i] = mykeys[m_i]
            m_i += 1
            t_i += 1
        else:
            temp_keys[t_i] = recvkeys[r_i]
            t_i += 1
            r_i += 1
    
    mykeys = temp_keys
    
    
    return mykeys

def MergeHigh(mykeys, recvkeys, sample_size):
    temp_keys = np.zeros((sample_size), dtype=np.intc)
    m_i = sample_size - 1
    r_i = sample_size - 1
    t_i = sample_size - 1

    while t_i >= 0:
        if mykeys[m_i] >= recvkeys[r_i]:
            temp_keys[t_i] = mykeys[m_i]
            m_i -= 1
            t_i -= 1
        else:
            temp_keys[t_i] = recvkeys[r_i]
            t_i -= 1
            r_i -= 1
    
    mykeys = temp_keys
    
    return mykeys

def ComputePartner(phase, myrank):
    partner = None
    if phase % 2 == 0:
        if myrank % 2 != 0:
            partner = myrank - 1
        else:
            partner = myrank + 1
    else:
        if myrank % 2 != 0:
            partner = myrank + 1
        else:
            partner = myrank - 1
    if partner == -1 or partner == size:
        return MPI.PROC_NULL
    return partner

def Get_send_args(loc_list, loc_n, which_splitter, islower):

    for i in range(loc_n):
        if loc_list[i] >= which_splitter:
            break
    if islower:
        return loc_n - i
    else: 
        return i

def Merge(loc_list, loc_n, rcv_buf, rcv_count): 
    l_i = 0
    r_i = 0
    t_i = 0
    tmp_array = np.zeros((loc_n + rcv_count), dtype=np.intc)

    while r_i < rcv_count and l_i < loc_n:
        
        if loc_list[l_i] <= rcv_buf[r_i]:
            tmp_array[t_i] = loc_list[l_i] 
            t_i += 1
            l_i += 1
        else:
            tmp_array[t_i] = rcv_buf[r_i]
            t_i += 1
            r_i += 1
    
    if(r_i == rcv_count):
        
        tmp_array = np.concatenate((tmp_array[0:t_i], loc_list[(l_i):(l_i + loc_n)]))
    
    else:
        tmp_array = np.concatenate((tmp_array[0:t_i], rcv_buf[r_i:rcv_count]))

    return tmp_array


###############input arrays###################

total_nums = 10000

local_n = int(total_nums/size)

my_sub_input = np.zeros((local_n), dtype=np.intc)

my_rcv_buf = np.zeros((total_nums), dtype=np.intc)

splitters = np.full((size), 0)

rcv_count_array = np.zeros((size), dtype=np.intc)

offset_array = np.zeros((size), dtype=np.intc)

sorted_array = np.zeros((total_nums), dtype=np.intc)

for i in range(local_n):
    my_sub_input[i] = random.randint(10000)

##############sample arrays####################
percentage = 0.01
sample_size = int(percentage * total_nums)
my_sub_sample = np.zeros((sample_size), dtype=np.intc)
rcv_keys = np.zeros((sample_size), dtype=np.intc)


for i in range(sample_size):
    my_sub_sample[i] = my_sub_input[i]

my_sub_sample = np.sort(my_sub_sample)



##############################Sorting Sample#####################################

for phase in range(size):
    partner = ComputePartner(phase, my_rank)

    if partner == MPI.PROC_NULL:
        continue    
    
    request = comm_world.Issend((my_sub_sample, sample_size, MPI.INT), dest=partner, tag=17)
    
    comm_world.Recv((rcv_keys, sample_size, MPI.INT), source=partner,  tag=17, status = status)

    status = MPI.Status()

    request.Wait(status)

    if my_rank < partner:
        my_sub_sample = MergeLow(my_sub_sample, rcv_keys, sample_size)
    
    else:
        my_sub_sample = MergeHigh(my_sub_sample, rcv_keys, sample_size)


##############################End of Sorting Sample#####################################


#################Splitters Calculation###################
my_min = my_sub_sample[0]
my_max = my_sub_sample[sample_size-1]
prev_max = 0
splitter = np.zeros((1), dtype=np.intc)
if(my_rank - 1 >= 0 and my_rank + 1 < size):

    request = comm_world.isend(my_max, dest = my_rank + 1, tag = 17)

    prev_max = comm_world.recv(source = my_rank - 1, tag = 17)

    request.Free()

    splitter[0] = math.ceil((my_min + prev_max)/2)
    

elif(my_rank - 1 < 0):
    comm_world.send(my_max, dest = my_rank + 1, tag = 17)


elif(my_rank + 1 > size - 1):
    prev_max = comm_world.recv(source = my_rank - 1, tag = 17)

    splitter[0] = math.ceil((my_min + prev_max)/2)

comm_world.Allgather(splitter, splitters)

splitters.resize(size + 1)

splitters[size] = 0

#################End of Splitters Calculation###################



#################Sample Sort Second Implementation###################

bitmask = int(size/2)
which_splitter = int(bitmask)

my_sub_input = np.sort(my_sub_input)

while(bitmask >= 1):
    partner =  int(my_rank) ^ int(bitmask)

    if(my_rank < partner):
        count = Get_send_args(my_sub_input, local_n, splitters[int(which_splitter)], True)

        new_loc_n = local_n - count

        offset = local_n - count

        bitmask /= 2

        which_splitter = int(which_splitter - bitmask)

    else:
        count = Get_send_args(my_sub_input, local_n, splitters[int(which_splitter)], False)

        new_loc_n = local_n - count

        offset = 0

        bitmask /=2

        which_splitter = int(which_splitter + bitmask)
    

    request = comm_world.Issend((my_sub_input[offset : offset + count], count, MPI.INT), dest=partner, tag=11)

    comm_world.Recv((my_rcv_buf, total_nums, MPI.INT), source=partner,  tag=11, status=status)

    rcv_count = status.Get_count(MPI.INT)

    status = MPI.Status()

    request.Wait(status)

    local_n = new_loc_n

    if my_rank < partner:
        offset = 0
    
    else:
        offset = count

    my_sub_input = Merge(my_sub_input[offset:(offset + local_n)], local_n, my_rcv_buf[0:rcv_count], rcv_count)

    local_n += rcv_count

end = MPI.Wtime()
######################End of Sample Sort Second Implementation########################


if(my_rank == 0):
    print(f"Total Runtime is: {end - start}")


##################Gather the sorted array into one final array################################
my_send_count = np.zeros((1), dtype=np.intc)

my_send_count[0] = my_sub_input.size



comm_world.Gather(my_send_count, rcv_count_array, root = 0)

if my_rank == 0:

    offset_array[0] = 0

    for i in range(1, size):       
        offset_array[i] = offset_array[i - 1] + rcv_count_array[i - 1]



comm_world.Gatherv(my_sub_input,[sorted_array, rcv_count_array, offset_array, MPI.INT], root = 0)

#if my_rank == 0:
   # print(f"I am P{my_rank} and final sorted array is: {sorted_array} and size = {sorted_array.size}")
    