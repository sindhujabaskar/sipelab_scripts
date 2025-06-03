
import os
import matplotlib.pyplot as plt
import nidaqmx
from nidaqmx.constants import (
    AcquisitionType,
    READ_ALL_AVAILABLE,
    LoggingMode,
    LoggingOperation,
)
from nptdms import TdmsFile

TDMS_FILE = nidaqmx.Task.in_stream.start_new_file(r'C:\dev\playground')

# remove old TDMS
for fn in (TDMS_FILE, TDMS_FILE + "_index"):
    if os.path.exists(fn):
        os.remove(fn)

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev2/ai0")
    task.timing.cfg_samp_clk_timing(
        1000.0, sample_mode=AcquisitionType.CONTINUOUS
    )
    task.timing.first_samp_timestamp_val
    # log incoming samples into TestData.tdms
    task.in_stream.configure_logging(
        TDMS_FILE,
        logging_mode=LoggingMode.LOG_AND_READ,
        operation=LoggingOperation.CREATE_OR_REPLACE,
    )
    task.start()
    print("Acquiring samples continuously. Press Ctrl+C to stop.")

    try:
        while True:
            _ = task.read(number_of_samples_per_channel=1000)
    except KeyboardInterrupt:
        pass
    finally:
        task.stop()

# read back and plot the logged TDMS data
with TdmsFile.open(TDMS_FILE) as tdms:
    for group in tdms.groups():
        for channel in group.channels():
            data = channel[:]
            print(f"{group.name}/{channel.name} → {len(data)} samples")
            plt.plot(data, label=channel.name)
    plt.legend()
    plt.show()


#  """Example for reading digital signals.

# This example demonstrates how to acquire a continuous digital
# waveform using the DAQ device's internal clock.
# """
# import matplotlib.pyplot as plt
# import nidaqmx
# from nidaqmx.constants import AcquisitionType

# with nidaqmx.Task() as task:
#     task.ai_channels.add_ai_voltage_chan("Dev2/ai0")
#     task.timing.cfg_samp_clk_timing(1000.0, sample_mode=AcquisitionType.CONTINUOUS)
#     task.start()
#     print("Acquiring samples continuously. Press Ctrl+C to stop.")

#     try:
#         total_read = 0
#         while True:
#             data = task.read(number_of_samples_per_channel=1000)
#             read = len(data)
#             total_read += read
#             print(f"Acquired data: {read} samples. Total {total_read}.", end="\r")
#     except KeyboardInterrupt:
#         pass
#     finally:
#         task.stop()
#         print(f"\nAcquired {total_read} total samples.")
#         print(data)
#         plt.plot(data)
#         plt.show()


# import os

# from nptdms import TdmsFile

# import nidaqmx
# from nidaqmx.constants import (
#     READ_ALL_AVAILABLE,
#     AcquisitionType,
#     LoggingMode,
#     LoggingOperation,
# )

# with nidaqmx.Task() as task:
#     task.ai_channels.add_ai_voltage_chan("Dev2/ai0")
#     task.timing.cfg_samp_clk_timing(1000.0, sample_mode=AcquisitionType.CONTINUOUS)
#     task.in_stream.configure_logging(
#         "TestData.tdms", LoggingMode.LOG_AND_READ, operation=LoggingOperation.CREATE_OR_REPLACE
#     )

#     task.read(READ_ALL_AVAILABLE)

# with TdmsFile.open("TestData.tdms") as tdms_file:
#     for group in tdms_file.groups():
#         for channel in group.channels():
#             data = channel[:]
#             print("Read data from TDMS file: [" + ", ".join(f"{value:f}" for value in data) + "]")

# if os.path.exists("TestData.tdms"):
#     os.remove("TestData.tdms")

# if os.path.exists("TestData.tdms_index"):
#     os.remove("TestData.tdms_index")