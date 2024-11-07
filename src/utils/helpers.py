import numpy as np
import math
import random
from scipy import signal


def random_start(sample_data, window_size, roll_sequence=False, sequence_part=1, num_sequence_parts=1):
    # Inspired by https://github.com/lshiwjx/2s-AGCN/blob/master/feeders/tools.py
    
    # Fetch sequence information
    C, T, V = sample_data.shape
    if not roll_sequence:
        try:
            T = np.where(np.isnan(sample_data[0, :, 0]))[0][0]
        except:
            pass
    
    # Select sequence starting point
    part_size = math.floor(T / num_sequence_parts)
    window_start_minimum = (sequence_part - 1) * part_size if (sequence_part - 1) * part_size < (T - window_size) else T - window_size
    window_start_maximum = sequence_part * part_size if (sequence_part * part_size) < (T - window_size) else T - window_size
    window_start = random.randint(window_start_minimum, window_start_maximum)
    
    return sample_data[:, window_start:window_start+window_size, :]


def rotate(sample_data, angle):
    C, T, V = sample_data.shape
    rad = math.radians(-angle)
    a = np.full(shape=T, fill_value=rad)
    theta = np.array([[np.cos(a), -np.sin(a)],  # Rotation matrix
                      [np.sin(a), np.cos(a)]])  # xuanzhuan juzhen

    # Rotate joints for each frame
    for i_frame in range(T):
        xy = sample_data[0:2, i_frame, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        sample_data[0:2, i_frame, :] = new_xy.reshape(2, V)
    return sample_data


def scale(sample_data, scale_factor):
    sample_data[0:2, :, :] = sample_data[0:2, :, :] * scale_factor
    return sample_data


def translation(sample_data, delta_x, delta_y):
    sample_data[0, :, :] = sample_data[0, :, :] + delta_x
    sample_data[1, :, :] = sample_data[1, :, :] + delta_y
    return sample_data


def random_perturbation(sample_data,
                angle_candidate=[i for i in range(-45, 45+1)],
                scale_candidate=[i/100 for i in range(int(0.7*100), int(1.3*100)+1)],
                translation_candidate=[i/100 for i in range(int(-0.3*100), int(0.3*100)+1)]):

    sample_data = rotate(sample_data, random.choice(angle_candidate))
    sample_data = scale(sample_data, random.choice(scale_candidate))
    sample_data = translation(sample_data, random.choice(translation_candidate), random.choice(translation_candidate))
    return sample_data


def local_position(global_position, bone_conns):
    local_position = np.zeros(global_position.shape)
    for i in range(len(bone_conns)):
        local_position[:,:,i,:] = global_position[:,:,i,:] - global_position[:,:,bone_conns[i],:]
    return local_position


def angular_position(linear_position):
    angular_position = np.zeros(tuple([2] + list(linear_position.shape[1:]))) # Currently limited to two dimensions
    rho = np.sqrt(linear_position[0,...]**2 + linear_position[1,...]**2)
    phi = np.arctan2(linear_position[1,...], linear_position[0,...])
    angular_position[0,...] = (rho-0.5)*2
    angular_position[1,...] = phi/math.pi
    return angular_position


def butterworth_filter(position, butterworth):
    C, T, V, M = position.shape
    filtered_position = np.zeros((C, T, V, M))
    for v in range(V):
        for c in range(C):
            filtered_position[c,:,v,0] = signal.sosfiltfilt(butterworth, position[c,:,v,0])
    return filtered_position
            

def derivative(variable, time_stride=1):
    C, T, V, M = variable.shape
    change = np.zeros((C, T, V, M))
    for t in range(T-time_stride):
        change[:,t,:,:] = (variable[:,t+time_stride,:,:] - variable[:,t,:,:])/time_stride
    return change


# Create motion features --> from Daniels code
# Inspired by https://github.com/yfsong0709/ResGCNv1/tree/master/src/dataset/data_utils.py
def create_motion_features(data, bone_conns, position_raw=False, position_filtered=True, velocity_raw=False,
                           velocity_filtered=True, acceleration_raw=False, acceleration_filtered=False,
                           linear_motion=True, angular_motion=False, global_motion=True, local_motion=False,
                           fast_velocity=True, slow_velocity=False, fast_acceleration=False, slow_acceleration=False,
                           butterworth_order=8):
    
    # Obtain data dimensions
    C, T, V, M = data.shape
    
    # Initialize tensor for motion features
    motion_features_set = []

    """Position features: [Raw linear global position X, Raw linear global position Y, Raw linear local position X, 
    Raw linear local position Y, Raw angular global position X, Raw angular global position Y, Raw angular local 
    position X, Raw angular local position Y, Filtered linear global position X, Filtered linear global position Y, 
    Filtered linear local position X, Filtered linear local position Y, Filtered angular global position X, 
    Filtered angular global position Y, Filtered angular local position X, Filtered angular local position Y]"""
    
    # Initialize tensor for position features
    position_features = []
    
    # Raw linear global position
    raw_linear_global_position = data
    if position_raw and linear_motion and global_motion:
        position_features.append(raw_linear_global_position)
        
    # Raw linear local position
    raw_linear_local_position = local_position(raw_linear_global_position, bone_conns)
    if position_raw and linear_motion and local_motion:
        position_features.append(raw_linear_local_position)
    
    # Raw angular global position
    raw_angular_global_position = angular_position(raw_linear_global_position)
    if position_raw and angular_motion and global_motion:        
        position_features.append(raw_angular_global_position) 
        
    # Raw angular local position
    raw_angular_local_position = angular_position(raw_linear_local_position)
    if position_raw and angular_motion and local_motion:        
        position_features.append(raw_angular_local_position) 
    
    # Butterworth zero-lag IIR filter (backward-forward filter)
    filter_order_limits = {1: 7, 2: 10, 4: 16, 8: 28, 16: 52, 32: 100}
    if T >= filter_order_limits[butterworth_order]:
        filter_order = butterworth_order
    else:
        filter_order = butterworth_order/2
        while True:
            if filter_order >= 1:
                if T >= filter_order_limits[filter_order]:
                    break
                else:
                    filter_order /= 2
            else:
                filter_order = None
                break
    if filter_order is not None:
        butterworth = signal.butter(int(filter_order), 0.5, output='sos')
                
    # Filtered linear global position
    filtered_linear_global_position = butterworth_filter(raw_linear_global_position, butterworth) \
        if filter_order is not None else raw_linear_global_position
    if position_filtered and linear_motion and global_motion:
        position_features.append(filtered_linear_global_position)
        
    # Filtered linear local position
    filtered_linear_local_position = butterworth_filter(raw_linear_local_position, butterworth) \
        if filter_order is not None else raw_linear_local_position
    if position_filtered and linear_motion and local_motion:
        position_features.append(filtered_linear_local_position)
    
    # Filtered angular global position
    filtered_angular_global_position = butterworth_filter(raw_angular_global_position, butterworth) \
        if filter_order is not None else raw_angular_global_position
    if position_filtered and angular_motion and global_motion:
        position_features.append(filtered_angular_global_position)
        
    # Filtered angular local position
    filtered_angular_local_position = butterworth_filter(raw_angular_local_position, butterworth) \
        if filter_order is not None else raw_angular_local_position
    if position_filtered and angular_motion and local_motion:
        position_features.append(filtered_angular_local_position)
    
    """
    Velocity features:
    [Raw linear global fast velocity X, Raw linear global fast velocity Y, Raw linear global slow velocity X, Raw linear global slow velocity Y,
    Raw linear local fast velocity X, Raw linear local fast velocity Y, Raw linear local slow velocity X, Raw linear local slow velocity Y,
    Raw angular global fast velocity X, Raw angular global fast velocity Y, Raw angular global slow velocity X, Raw angular global slow velocity Y,
    Raw angular local fast velocity X, Raw angular local fast velocity Y, Raw angular local slow velocity X, Raw angular local slow velocity Y,
    Filtered linear global fast velocity X, Filtered linear global fast velocity Y, Filtered linear global slow velocity X, Filtered linear global slow velocity Y,
    Filtered linear local fast velocity X, Filtered linear local fast velocity Y, Filtered linear local slow velocity X, Filtered linear local slow velocity Y,
    Filtered angular global fast velocity X, Filtered angular global fast velocity Y, Filtered angular global slow velocity X, Filtered angular global slow velocity Y,
    Filtered angular local fast velocity X, Filtered angular local fast velocity Y, Filtered angular local slow velocity X, Filtered angular local slow velocity Y]
    """
    
    # Initialize tensor for velocity features
    velocity_features = []
    
    # Raw linear global fast velocity
    raw_linear_global_fast_velocity = derivative(raw_linear_global_position, time_stride=1)
    if velocity_raw and linear_motion and global_motion and fast_velocity:
        velocity_features.append(raw_linear_global_fast_velocity)
    
    # Raw linear global slow velocity
    raw_linear_global_slow_velocity = derivative(raw_linear_global_position, time_stride=2)
    if velocity_raw and linear_motion and global_motion and slow_velocity:
        velocity_features.append(raw_linear_global_slow_velocity)
    
    # Raw linear local fast velocity
    raw_linear_local_fast_velocity = derivative(raw_linear_local_position, time_stride=1)
    if velocity_raw and linear_motion and local_motion and fast_velocity:
        velocity_features.append(raw_linear_local_fast_velocity)
        
    # Raw linear local slow velocity
    raw_linear_local_slow_velocity = derivative(raw_linear_local_position, time_stride=2)
    if velocity_raw and linear_motion and local_motion and slow_velocity:
        velocity_features.append(raw_linear_local_slow_velocity)
    
    # Raw angular global fast velocity
    raw_angular_global_fast_velocity = derivative(raw_angular_global_position, time_stride=1)
    if velocity_raw and angular_motion and global_motion and fast_velocity:
        velocity_features.append(raw_angular_global_fast_velocity)
    
    # Raw angular global slow velocity
    raw_angular_global_slow_velocity = derivative(raw_angular_global_position, time_stride=2)
    if velocity_raw and angular_motion and global_motion and slow_velocity:
        velocity_features.append(raw_angular_global_slow_velocity)
        
    # Raw angular local fast velocity
    raw_angular_local_fast_velocity = derivative(raw_angular_local_position, time_stride=1)
    if velocity_raw and angular_motion and local_motion and fast_velocity:
        velocity_features.append(raw_angular_local_fast_velocity)
    
    # Raw angular local slow velocity
    raw_angular_local_slow_velocity = derivative(raw_angular_local_position, time_stride=2)
    if velocity_raw and angular_motion and local_motion and slow_velocity:
        velocity_features.append(raw_angular_local_slow_velocity)
    
    # Filtered linear global fast velocity
    filtered_linear_global_fast_velocity = derivative(filtered_linear_global_position, time_stride=1)
    if velocity_filtered and linear_motion and global_motion and fast_velocity:
        velocity_features.append(filtered_linear_global_fast_velocity)
    
    # Filtered linear global slow velocity
    filtered_linear_global_slow_velocity = derivative(filtered_linear_global_position, time_stride=2)
    if velocity_filtered and linear_motion and global_motion and slow_velocity:
        velocity_features.append(filtered_linear_global_slow_velocity)
        
    # Filtered linear local fast velocity
    filtered_linear_local_fast_velocity = derivative(filtered_linear_local_position, time_stride=1)
    if velocity_filtered and linear_motion and local_motion and fast_velocity:
        velocity_features.append(filtered_linear_local_fast_velocity)
    
    # Filtered linear local slow velocity
    filtered_linear_local_slow_velocity = derivative(filtered_linear_local_position, time_stride=2)
    if velocity_filtered and linear_motion and local_motion and slow_velocity:
        velocity_features.append(filtered_linear_local_slow_velocity)
    
    # Filtered angular global fast velocity
    filtered_angular_global_fast_velocity = derivative(filtered_angular_global_position, time_stride=1)
    if velocity_filtered and angular_motion and global_motion and fast_velocity:
        velocity_features.append(filtered_angular_global_fast_velocity)
    
    # Filtered angular global slow velocity
    filtered_angular_global_slow_velocity = derivative(filtered_angular_global_position, time_stride=2)
    if velocity_filtered and angular_motion and global_motion and slow_velocity:
        velocity_features.append(filtered_angular_global_slow_velocity)
        
    # Filtered angular local fast velocity
    filtered_angular_local_fast_velocity = derivative(filtered_angular_local_position, time_stride=1)
    if velocity_filtered and angular_motion and local_motion and fast_velocity:
        velocity_features.append(filtered_angular_local_fast_velocity)
    
    # Filtered angular local slow velocity
    filtered_angular_local_slow_velocity = derivative(filtered_angular_local_position, time_stride=2)
    if velocity_filtered and angular_motion and local_motion and slow_velocity:
        velocity_features.append(filtered_angular_local_slow_velocity)
    
    """
    Acceleration features:
    [Raw linear global fast-fast acceleration X, Raw linear global fast-fast acceleration Y, Raw linear global fast-slow acceleration X, Raw linear global fast-slow acceleration Y,
    Raw linear global slow-fast acceleration X, Raw linear global slow-fast acceleration Y, Raw linear global slow-slow acceleration X, Raw linear global slow-slow acceleration Y,
    Raw linear local fast-fast acceleration X, Raw linear local fast-fast acceleration Y, Raw linear local fast-slow acceleration X, Raw linear local fast-slow acceleration Y,
    Raw linear local slow-fast acceleration X, Raw linear local slow-fast acceleration Y, Raw linear local slow-slow acceleration X, Raw linear local slow-slow acceleration Y,
    Raw angular global fast-fast acceleration X, Raw angular global fast-fast acceleration Y, Raw angular global fast-slow acceleration X, Raw angular global fast-slow acceleration Y,
    Raw angular global slow-fast acceleration X, Raw angular global slow-fast acceleration Y, Raw angular global slow-slow acceleration X, Raw angular global slow-slow acceleration Y,
    Raw angular local fast-fast acceleration X, Raw angular local fast-fast acceleration Y, Raw angular local fast-slow acceleration X, Raw angular local fast-slow acceleration Y,
    Raw angular local slow-fast acceleration X, Raw angular local slow-fast acceleration Y, Raw angular local slow-slow acceleration X, Raw angular local slow-slow acceleration Y,
    Filtered linear global fast-fast acceleration X, Filtered linear global fast-fast acceleration Y, Filtered linear global fast-slow acceleration X, Filtered linear global fast-slow acceleration Y,
    Filtered linear global slow-fast acceleration X, Filtered linear global slow-fast acceleration Y, Filtered linear global slow-slow acceleration X, Filtered linear global slow-slow acceleration Y,
    Filtered linear local fast-fast acceleration X, Filtered linear local fast-fast acceleration Y, Filtered linear local fast-slow acceleration X, Filtered linear local fast-slow acceleration Y,
    Filtered linear local slow-fast acceleration X, Filtered linear local slow-fast acceleration Y, Filtered linear local slow-slow acceleration X, Filtered linear local slow-slow acceleration Y,
    Filtered angular global fast-fast acceleration X, Filtered angular global fast-fast acceleration Y, Filtered angular global fast-slow acceleration X, Filtered angular global fast-slow acceleration Y,
    Filtered angular global slow-fast acceleration X, Filtered angular global slow-fast acceleration Y, Filtered angular global slow-slow acceleration X, Filtered angular global slow-slow acceleration Y,
    Filtered angular local fast-fast acceleration X, Filtered angular local fast-fast acceleration Y, Filtered angular local fast-slow acceleration X, Filtered angular local fast-slow acceleration Y,
    Filtered angular local slow-fast acceleration X, Filtered angular local slow-fast acceleration Y, Filtered angular local slow-slow acceleration X, Filtered angular local slow-slow acceleration Y]
    """
    
    # Initialize tensor for acceleration features
    acceleration_features = []
    
    # Raw linear global fast-fast acceleration
    if acceleration_raw and linear_motion and global_motion and fast_velocity and fast_acceleration:
        acceleration_features.append(derivative(raw_linear_global_fast_velocity, time_stride=1))
    
    # Raw linear global fast-slow acceleration
    if acceleration_raw and linear_motion and global_motion and fast_velocity and slow_acceleration:
        acceleration_features.append(derivative(raw_linear_global_fast_velocity, time_stride=2))
    
    # Raw linear global slow-fast acceleration
    if acceleration_raw and linear_motion and global_motion and slow_velocity and fast_acceleration:
        acceleration_features.append(derivative(raw_linear_global_slow_velocity, time_stride=1))
    
    # Raw linear global slow-slow acceleration
    if acceleration_raw and linear_motion and global_motion and slow_velocity and slow_acceleration:
        acceleration_features.append(derivative(raw_linear_global_slow_velocity, time_stride=2))
        
    # Raw linear local fast-fast acceleration
    if acceleration_raw and linear_motion and local_motion and fast_velocity and fast_acceleration:
        acceleration_features.append(derivative(raw_linear_local_fast_velocity, time_stride=1))
    
    # Raw linear local fast-slow acceleration
    if acceleration_raw and linear_motion and local_motion and fast_velocity and slow_acceleration:
        acceleration_features.append(derivative(raw_linear_local_fast_velocity, time_stride=2))
    
    # Raw linear local slow-fast acceleration
    if acceleration_raw and linear_motion and local_motion and slow_velocity and fast_acceleration:
        acceleration_features.append(derivative(raw_linear_local_slow_velocity, time_stride=1))
    
    # Raw linear local slow-slow acceleration
    if acceleration_raw and linear_motion and local_motion and slow_velocity and slow_acceleration:
        acceleration_features.append(derivative(raw_linear_local_slow_velocity, time_stride=2))
    
    # Raw angular global fast-fast acceleration
    if acceleration_raw and angular_motion and global_motion and fast_velocity and fast_acceleration:
        acceleration_features.append(derivative(raw_angular_global_fast_velocity, time_stride=1))
    
    # Raw angular global fast-slow acceleration
    if acceleration_raw and angular_motion and global_motion and fast_velocity and slow_acceleration:
        acceleration_features.append(derivative(raw_angular_global_fast_velocity, time_stride=2))
    
    # Raw angular global slow-fast acceleration
    if acceleration_raw and angular_motion and global_motion and slow_velocity and fast_acceleration:
        acceleration_features.append(derivative(raw_angular_global_slow_velocity, time_stride=1))
    
    # Raw angular global slow-slow acceleration
    if acceleration_raw and angular_motion and global_motion and slow_velocity and slow_acceleration:
        acceleration_features.append(derivative(raw_angular_global_slow_velocity, time_stride=2))
        
    # Raw angular local fast-fast acceleration
    if acceleration_raw and angular_motion and local_motion and fast_velocity and fast_acceleration:
        acceleration_features.append(derivative(raw_angular_local_fast_velocity, time_stride=1))
    
    # Raw angular local fast-slow acceleration
    if acceleration_raw and angular_motion and local_motion and fast_velocity and slow_acceleration:
        acceleration_features.append(derivative(raw_angular_local_fast_velocity, time_stride=2))
    
    # Raw angular local slow-fast acceleration
    if acceleration_raw and angular_motion and local_motion and slow_velocity and fast_acceleration:
        acceleration_features.append(derivative(raw_angular_local_slow_velocity, time_stride=1))
    
    # Raw angular local slow-slow acceleration
    if acceleration_raw and angular_motion and local_motion and slow_velocity and slow_acceleration:
        acceleration_features.append(derivative(raw_angular_local_slow_velocity, time_stride=2))

    # Filtered linear global fast-fast acceleration
    if acceleration_filtered and linear_motion and global_motion and fast_velocity and fast_acceleration:
        acceleration_features.append(derivative(filtered_linear_global_fast_velocity, time_stride=1))
    
    # Filtered linear global fast-slow acceleration
    if acceleration_filtered and linear_motion and global_motion and fast_velocity and slow_acceleration:
        acceleration_features.append(derivative(filtered_linear_global_fast_velocity, time_stride=2))
    
    # Filtered linear global slow-fast acceleration
    if acceleration_filtered and linear_motion and global_motion and slow_velocity and fast_acceleration:
        acceleration_features.append(derivative(filtered_linear_global_slow_velocity, time_stride=1))
    
    # Filtered linear global slow-slow acceleration
    if acceleration_filtered and linear_motion and global_motion and slow_velocity and slow_acceleration:
        acceleration_features.append(derivative(filtered_linear_global_slow_velocity, time_stride=2))
        
    # Filtered linear local fast-fast acceleration
    if acceleration_filtered and linear_motion and local_motion and fast_velocity and fast_acceleration:
        acceleration_features.append(derivative(filtered_linear_local_fast_velocity, time_stride=1))
    
    # Filtered linear local fast-slow acceleration
    if acceleration_filtered and linear_motion and local_motion and fast_velocity and slow_acceleration:
        acceleration_features.append(derivative(filtered_linear_local_fast_velocity, time_stride=2))
    
    # Filtered linear local slow-fast acceleration
    if acceleration_filtered and linear_motion and local_motion and slow_velocity and fast_acceleration:
        acceleration_features.append(derivative(filtered_linear_local_slow_velocity, time_stride=1))
    
    # Filtered linear local slow-slow acceleration
    if acceleration_filtered and linear_motion and local_motion and slow_velocity and slow_acceleration:
        acceleration_features.append(derivative(filtered_linear_local_slow_velocity, time_stride=2))
    
    # Filtered angular global fast-fast acceleration
    if acceleration_filtered and angular_motion and global_motion and fast_velocity and fast_acceleration:
        acceleration_features.append(derivative(filtered_angular_global_fast_velocity, time_stride=1))
    
    # Filtered angular global fast-slow acceleration
    if acceleration_filtered and angular_motion and global_motion and fast_velocity and slow_acceleration:
        acceleration_features.append(derivative(filtered_angular_global_fast_velocity, time_stride=2))
    
    # Filtered angular global slow-fast acceleration
    if acceleration_filtered and angular_motion and global_motion and slow_velocity and fast_acceleration:
        acceleration_features.append(derivative(filtered_angular_global_slow_velocity, time_stride=1))
    
    # Filtered angular global slow-slow acceleration
    if acceleration_filtered and angular_motion and global_motion and slow_velocity and slow_acceleration:
        acceleration_features.append(derivative(filtered_angular_global_slow_velocity, time_stride=2))
    
    # Filtered angular local fast-fast acceleration
    if acceleration_filtered and angular_motion and local_motion and fast_velocity and fast_acceleration:
        acceleration_features.append(derivative(filtered_angular_local_fast_velocity, time_stride=1))
    
    # Filtered angular local fast-slow acceleration
    if acceleration_filtered and angular_motion and local_motion and fast_velocity and slow_acceleration:
        acceleration_features.append(derivative(filtered_angular_local_fast_velocity, time_stride=2))
    
    # Filtered angular local slow-fast acceleration
    if acceleration_filtered and angular_motion and local_motion and slow_velocity and fast_acceleration:
        acceleration_features.append(derivative(filtered_angular_local_slow_velocity, time_stride=1))
    
    # Filtered angular local slow-slow acceleration
    if acceleration_filtered and angular_motion and local_motion and slow_velocity and slow_acceleration:
        acceleration_features.append(derivative(filtered_angular_local_slow_velocity, time_stride=2))

    # Concatenate motion feature tensors
    num_position_features = len(position_features)
    num_velocity_features = len(velocity_features)
    num_acceleration_features = len(acceleration_features)
    F = num_position_features + num_velocity_features + num_acceleration_features
    motion_features = np.zeros((F, C, T, V))
    index = 0
    for i in range(num_position_features):
        motion_features[index,:,:,:] = np.asarray(position_features[i][...,0])
        index += 1
    for i in range(num_velocity_features):
        motion_features[index,:,:,:] = np.asarray(velocity_features[i][...,0])
        index += 1
    for i in range(num_acceleration_features):
        motion_features[index,:,:,:] = np.asarray(acceleration_features[i][...,0])
        index += 1
    F, C, T, V = motion_features.shape
    motion_features = np.reshape(motion_features, (F*C, T, V))
        
    return motion_features