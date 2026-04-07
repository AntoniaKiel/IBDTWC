# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 17:31:52 2025

@author: anton
"""
from obspy import UTCDateTime
#from obspy.clients.fdsn import Client
from obspy import read
from obspy.clients.fdsn import RoutingClient
import numpy as np
from scipy.signal import medfilt
from datetime import datetime, timedelta, timezone
import logging
from obspy.clients.fdsn import Client
from concurrent.futures import ThreadPoolExecutor
from obspy import read_inventory, UTCDateTime


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="obspy")

from src.data_download.preprocessing_utils import windowed_rms

def read_Neumayer_data(station_list,t_start,t_end,fmin,fmax,FROM_DRIVE=False,pre_proc=False,show_data=False):
    
    if FROM_DRIVE:
        import glob
        from pyrocko import io, trace
        
        # read list of station names
        sta_names=np.load('../../DATA/stations_list.npy',allow_pickle=True)
        sta_names[1],sta_names[2]=sta_names[2],sta_names[1]
        #set parameters
        year=2004
        day=2
        station_idx=1
        
        
        
        # read seismic data of that day to array
        file_path='../../DATA/NEUMAYER/nm-silo-2004/mseed_2004_'+str(day).zfill(3)
        files_path_all=glob.glob(file_path+'/*_2004_'+str(day).zfill(3)+'.BHZ')
        n_sta=len(files_path_all)
        
        DATA_times=list()
        DATA_DAY_list=list()
        for idxSta in range(n_sta):
            logging.info('working on ',files_path_all[idxSta],flush=True)
            
            traces=io.load(files_path_all[idxSta])
            traces_new=trace.degapper(traces,maxgap=3)
            
            times_new=list()
            for idx in range(len(traces_new)):
                traces_new[idx].bandpass(4,fmin,fmax)
                
                times_part=[datetime.fromtimestamp(timestamp,tz=timezone.utc) for timestamp in traces_new[idx].get_xdata()]
                times_new=times_new+times_part

            
            DATA_DAY_list.append(traces_new[0].ydata)
            DATA_times.append(times_new)      
        sampling=DATA_times[0][1]-DATA_times[0][0]   
        
        # cut data to desired time window used in connys paper (but longer for LTA trigger calculation)
        event_time=datetime(2004,1,2,4,32,30,tzinfo=timezone.utc)
        
        # window used to test STA/LTA trigger
        start_time=datetime(2004,1,2,4,10,30,tzinfo=timezone.utc)
        duration=40*60 # in seconds    
        
        time=DATA_times[station_idx][find_closest_datetime_index(DATA_times[0],start_time):find_closest_datetime_index(DATA_times[0],start_time+timedelta(seconds=duration))]    
        y=DATA_DAY_list[station_idx][find_closest_datetime_index(DATA_times[0],start_time):find_closest_datetime_index(DATA_times[0],start_time+timedelta(seconds=duration))] 

        # cut data to desired time window used in connys paper (but longer for LTA trigger calculation)
        event_time=datetime(2004,1,2,4,32,30,tzinfo=timezone.utc)
        
        # window used to test STA/LTA trigger
        start_time=datetime(2004,1,2,4,10,30,tzinfo=timezone.utc)
        duration=40*60 # in seconds    
        
        time=DATA_times[station_idx][find_closest_datetime_index(DATA_times[0],start_time):find_closest_datetime_index(DATA_times[0],start_time+timedelta(seconds=duration))]    
        y=DATA_DAY_list[station_idx][find_closest_datetime_index(DATA_times[0],start_time):find_closest_datetime_index(DATA_times[0],start_time+timedelta(seconds=duration))] 
    
    
    # READ DATA DIRECTLY FROM SERVER
    else:
        from obspy import read_inventory
        import matplotlib.pyplot as plt
        
        from get_new_data_GFZ import test_data_fetch
        
        st=test_data_fetch('AW', station_list, t_start , t_end)
        try:
            st.merge(method=1, fill_value=0)
        except Exception as e:
            logging.info("Error merging traces",e)
        # if not st:
        #     sys.exit('No traces found! Skipping...')
        # elif len(st) != 16:
        #     sys.exit('Some error with number of traces, skipping...')
        if not st:
            logging.info('No traces found! Skipping...')
            
            return([],[],[],[])
        elif len(st) != 16:
            logging.info('Some error with number of traces, skipping...')
            return([],[],[],[])
        else:
            logging.debug('16 traces after merging...')
            inv=read_inventory('./inventory.xml')        
            # PRE-PROCESSING of data
            if pre_proc == True:
            # remove instrument response
                
                pre_filt=[0.001, 0.005, 45, 50]
                show_data=False
                output_type='VEL'
                
                st.detrend('linear')
                st.detrend('demean')
                st.taper(max_percentage=0.05,type=hann)
                for idx, tr in enumerate(st):
                    tr.remove_response(inventory=inv,output=output_type,pre_filt=pre_filt,water_level=60)

                    
            else:
                output_type='counts'
                    
            if show_data:
                # plot data
                fig,ax=plt.subplots(len(st),1,sharex=True,figsize=(8,5),dpi=300)
    
            
            st.filter('bandpass',freqmin=fmin,freqmax=fmax,corners=4,zerophase=True)
            # print('filtered...')
            
            if show_data:
                fig.autofmt_xdate()
                plt.show()
            
            
            # if len(st) != 1:
            #     print('len of all traces is equal:', len(st[0].data)==len(st[1].data))
            
            
            # Step 2: Define a common time vector based on desired start, end, and sampling rate
            start_time=np.min([tr.times("utcdatetime")[0] for tr in st])
            end_time=np.max([tr.times("utcdatetime")[-1] for tr in st])
            
            
            common_start_time = UTCDateTime(start_time)
            common_end_time = UTCDateTime(end_time)
            sampling_rate = 50.0  #  in Hz
            
            # Calculate the number of samples for the common time vector
            n_samples = int((common_end_time - common_start_time) * sampling_rate)
            common_time_vector = pd.date_range(start=start_time.datetime, end=end_time.datetime, periods=n_samples)
    
            
            # Step 3: Resample or align each trace in the stream to the common time vector
            aligned_traces = []
            station_coord=np.zeros((2,len(st)))
            
            for idx_tr,trace in enumerate(st):
                logging.info('start processing for trace ',idx_tr)
                # Create a time vector for this trace
                trace_time_vector = pd.to_datetime(np.linspace(trace.stats.starttime.timestamp, 
                                                           trace.stats.endtime.timestamp, trace.stats.npts), unit='s', utc=True)
                # Interpolate to the common time vector and fill missing values with NaN
                resampled_trace = np.interp(common_time_vector, trace_time_vector, trace.data, left=np.nan, right=np.nan)
                aligned_traces.append(resampled_trace)
                
                station_coord[0][idx_tr]=inv.get_coordinates(trace.get_id())['longitude']
                station_coord[1][idx_tr]=inv.get_coordinates(trace.get_id())['latitude']
            # Step 4: Stack the aligned traces into a 2D array (shape: number of stations, length of trace)
            seismic_data = np.vstack(aligned_traces)
    


        
            
        return(st,seismic_data,common_time_vector,output_type,station_coord)
    
def process_trace(
        trace,
        common_time_vector,
        station_coords_dict,
        inventory=None,
        fmin=0.1,
        fmax=20.0,
        clipping_factor=4,
        response_output=None,
        rms_window=100
    ):


    # Safety: if pre_proc=True but inventory not provided → raise error
    if response_output is not None and inventory is None:
        raise ValueError("inventory must be provided when pre_proc=True")

    # -----------------------------------------------------------
    # 1) OPTIONAL: pre-processing (detrend, taper, remove response)
    # -----------------------------------------------------------
    if response_output is not None:
        pre_filt = [0.001, 0.005, 45, 50]
        from scipy.signal.windows import hann
        
        trace.detrend("linear")
        trace.detrend("demean")
        trace.taper(max_percentage=0.05, type="hann")

        trace.remove_response(
            inventory=inventory,
            output=response_output,
            pre_filt=pre_filt,
            water_level=60
        )

    # -----------------------------------------------------------
    # 2) ALWAYS: bandpass filter
    # -----------------------------------------------------------
    trace.filter(
        "bandpass",
        freqmin=fmin,
        freqmax=fmax,
        corners=4,
        zerophase=True
    )

    # -----------------------------------------------------------
    # Adaptive clipping + median filter
    # ------------------------
    data = trace.data.astype(float)

    # RMS für Clipping
    amp = windowed_rms(data, rms_window, sampling_rate=trace.stats.sampling_rate)
    mean_amp = np.mean(amp)
    std_amp = np.std(amp)
    threshold = mean_amp + clipping_factor * std_amp

    # Medianfilter zuerst
    median_window_size = 5  # ungerade Zahl
    data_median = medfilt(data, kernel_size=median_window_size)

    # Clipping
    #data_clipped = np.clip(data_median, -threshold, threshold)
    data_clipped = np.tanh(data / threshold) * threshold

    trace.data = data_clipped  # update trace

    # ------------------------
    # Thread-safe debug-Plot (OOP API)
    # ------------------------
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    fig = Figure(figsize=(6, 4))
    canvas = FigureCanvas(fig)
    axes = fig.subplots(3, 1)

    axes[0].plot(data)
    axes[0].set_title("raw")
    axes[1].plot(data_median)
    axes[1].set_title("median filtered")
    axes[2].plot(data_clipped)
    axes[2].set_title("clipped")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f'./raw_traces_{trace.id}_{timestamp}.png'
    fig.savefig(filename)
    fig.clf()  # Figure freigeben

    # -----------------------------------------------------------
    # 4) Resample to common time vector
    # -----------------------------------------------------------
    trace_time_vector = np.linspace(
        trace.stats.starttime.timestamp,
        trace.stats.endtime.timestamp,
        trace.stats.npts
    )

    resampled_trace = np.interp(
        common_time_vector,
        trace_time_vector,
        trace.data,
        left=np.nan,
        right=np.nan
    )

    # -----------------------------------------------------------
    # 5) Coordinates
    # -----------------------------------------------------------
    coords = station_coords_dict[trace.get_id()]
    lon = coords["longitude"]
    lat = coords["latitude"]

    return resampled_trace, lon, lat

def process_seismic_data(station_list, t_start, t_end, fmin=1, fmax=10, clipping=4, response_output=None, FROM_DRIVE=False, pre_proc=False,
                         rms_window=100, show_data=False, sampling_rate=50.0):

    st = get_raw_data('AW', station_list, t_start, t_end)
    if st is None:
        raise ConnectionError("Failed to fetch data. Check your internet connection.")
    try:
        st.merge(method=1, fill_value=0)
    except Exception as e:
        logging.info(f"Error merging traces {e}")
    logging.debug(f"{len(st)} traces after merging")

    if not st:
        logging.info('No traces found! Skipping...')
        return [], []
    if len(st) != 16:
        logging.info('Some error with number of traces, skipping...')
        return [], []

    inv = read_inventory('./inventory.xml')
    t_start = UTCDateTime(t_start)
    t_end = UTCDateTime(t_end)
    n_samples = int((t_end - t_start) * sampling_rate)
    common_time_vector = np.linspace(t_start.timestamp, t_end.timestamp, n_samples)

    # if pre_proc:
    #     pre_filt = [0.001, 0.005, 45, 50]
    #     st.detrend('linear')
    #     st.detrend('demean')
    #     st.taper(max_percentage=0.05, type='hann')
    #     for tr in st:
    #         tr.remove_response(inventory=inv, output='VEL', pre_filt=pre_filt, water_level=60)

    # st.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)

    station_coords_dict = {tr.get_id(): inv.get_coordinates(tr.get_id()) for tr in st}
    aligned_traces = np.empty((len(st), n_samples))
    station_coords = np.zeros((2, len(st)))



    with ThreadPoolExecutor() as executor:
        results = list(executor.map(
            lambda tr: process_trace(
                tr,
                common_time_vector,
                station_coords_dict,         
                inventory=inv,          
                fmin=fmin, fmax=fmax,    
                clipping_factor=clipping,
                response_output=response_output,
                rms_window=rms_window
            ),
            st
        ))

    for idx, (resampled_trace, lon, lat) in enumerate(results):
        aligned_traces[idx, :] = resampled_trace
        station_coords[0, idx] = lon
        station_coords[1, idx] = lat

    if show_data:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(st), 1, sharex=True, figsize=(8, 5), dpi=300)
        for i, trace_data in enumerate(aligned_traces):
            ax[i].plot(common_time_vector, trace_data, label=st[i].id)
            ax[i].legend(loc="upper right")
        fig.autofmt_xdate()
        plt.show()

    return aligned_traces, common_time_vector


def get_available_stations(network_code: str, t_start: str, t_end: str) -> list[str]:
    """
    Retrieve a list of available stations for a given network and time window.

    Parameters
    ----------
    network_code : str
        The network code to query (e.g., 'GR', 'IU').
    t_start : str
        Start time in ISO 8601 format, e.g., '2022-01-12T00:00:00+00:00'.
    t_end : str
        End time in ISO 8601 format, e.g., '2022-01-12T23:59:59+00:00'.

    Returns
    -------
    list[str]
        List of station codes available for the specified network and time window.
        Returns an empty list if no stations are found.

    Notes
    -----
    - The function converts input times to UTC and formats them with 'Z' for ObsPy RoutingClient.
    - Logs warnings if no networks are available for the requested time window.
    """
    station_list = []

    # Initialize the routing client
    client = Client("GFZ")

    # turn into obspy UTCDateTime format to extract events
    t_start_utc = UTCDateTime(t_start)
    t_end_utc   = UTCDateTime(t_end)


    # Query the routing client
    inventory = client.get_stations(
        network=network_code,
        station='*',
        starttime=t_start_utc,
        endtime=t_end_utc,
        level='station'
    )

    logging.debug(f"Inventory of stations is {inventory}")
    # Check if any networks are returned
    if not inventory.networks:
        logging.warning(f"No networks available for {t_start_str[:-1]} - {t_end_str[:-1]}")
        return []

    # Extract station codes from inventory
    for network in inventory.networks:
        for station in network.stations:
            contents = station.get_contents()
            station_code = contents['stations'][0].split()[0]
            station_list.append(station_code)

    logging.debug("Got available stations sucessfully!")
    return station_list

def sort_stream_by_zne(stream):
    # Define a custom order for channels: Z -> N -> E
    channel_order = {'Z': 0, 'N': 1, 'E': 2}
    
    # Define a sorting key function
    def channel_sort_key(trace):
        # Get the last character of the channel (Z, N, E, etc.)
        channel_suffix = trace.stats.channel[-1]  # e.g., BH**Z**, BH**N**, BH**E**
        # Return the order value (0 for Z, 1 for N, 2 for E), or a high value for others
        return channel_order.get(channel_suffix, 999)  # Unknown channels get sorted last

    # Use sorted() instead of stream.sort() since Stream doesn't support key argument
    sorted_stream = sorted(stream, key=channel_sort_key)
    
    # Convert the list of sorted traces back into a Stream object
    return stream.__class__(sorted_stream)

def get_raw_data(network_code:str,
                     station_list: list, 
                     start_time:str, end_time:str):
    """This function read raw seismic data of watzmann array and returns the Z-component for every station as an obspy stream

    Parameters
    ----------
    network_code : str
        network code of desired array (here "AW")
    station_list : list
        list of station names to use for data extraction
    start_time : str
        Start time as an ISO 8601 timestamp string, including timezone offset.
        Expected format: 'YYYY-MM-DDTHH:MM:SS±HH:MM', e.g. '2022-01-12T00:00:00+00:00'.
    end_time : str
        End time as an ISO 8601 timestamp string, including timezone offset.
        Expected format: 'YYYY-MM-DDTHH:MM:SS±HH:MM', e.g. '2022-01-12T23:59:59+00:00'.

    Return
    ---------
    obspy.stream
        it returns the obspy stream of desired time window for every station and the Z component
    """


    logging.debug(f"start getting the raw data from server from {start_time} until {end_time}!")
    # check if token is still valid
    # token_path=Path('./.eidatoken')
    # if token_path.exists():
    #     # get timestamp of .eidatoken file
    #     # keep working with UTC timezone in case server run using UTC timezone -> ensurred correct timing
    #     mtime = datetime.fromtimestamp(token_path.stat().st_mtime, tz=timezone.utc)
    
    #     # calculte cutoff time (30 days ago -> go to 29 to ensure calculation neither takes longer than 30 day boundary)
    #     cutoff = datetime.now(timezone.utc) - timedelta(days=29)
    
    #     if mtime < cutoff:
    #         raise RuntimeError(f"{token_path} is older than 30 days (changed on {mtime}). Script run is stopped.")
    #     else:
    #         logging.info(f"{token_path} is newer or exactly 30 days old (changed on {mtime}).")
    
    #client = RoutingClient("eida-routing", credentials={'EIDA_TOKEN': './.eidatoken'})
    client = Client("GFZ")

    try:
        for idx,station_id in enumerate(station_list):
            # define channel name because its different for the 3-component station
            if station_id in ["VNA1", "VNA2", "VNA3"]:
                channel_id='BHZ'
            else:
                channel_id='SHZ'            
            
            if idx==0:
                stream = client.get_waveforms(network=network_code, 
                                               station=station_id, 
                                               location='*', 
                                               channel=channel_id,  
                                               starttime=UTCDateTime(start_time), 
                                               endtime=UTCDateTime(end_time))
            else:
                stream +=client.get_waveforms(network=network_code, 
                                               station=station_id,  
                                               location='*', 
                                               channel=channel_id,  
                                               starttime=UTCDateTime(start_time), 
                                               endtime=UTCDateTime(end_time))
        
        logging.debug(f"Fetched {len(stream)} traces for {station_list} from {start_time} to {end_time}.")
            
            
        
        return(stream)
    except Exception as e:
        logging.info(f"Error fetching data for {station_id}: {e}")
        
    








