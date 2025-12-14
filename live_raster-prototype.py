import argparse
import queue
import threading
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import serial


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot incoming serial numbers in real time.")
    parser.add_argument("--port", default="COM5", help="Serial port name (default: COM5).")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate (default: 115200).")
    parser.add_argument("--timeout", type=float, default=0.1, help="Serial read timeout in seconds (default: 0.1).")
    parser.add_argument("--window", type=int, default=500, help="Number of points to keep on screen (default: 500).")
    parser.add_argument("--interval", type=int, default=50, help="Plot refresh interval in milliseconds (default: 50).")
    return parser.parse_args()


def serial_reader(ser: serial.Serial, target: queue.Queue, stop_event: threading.Event) -> None:
    """Continuously read lines, convert them to floats, and push them into the queue."""

    while not stop_event.is_set():
        try:
            raw = ser.readline()
        except serial.SerialException as exc:
            print(f"Serial error: {exc}")
            break

        if not raw:
            continue

        text = raw.decode("utf-8", errors="ignore").strip()
        if not text:
            continue

        parts = text.replace(",", " ").split()
        try:
            values = [float(part) for part in parts]
        except ValueError:
            continue

        if values:
            target.put((time.perf_counter(), values))


def main() -> int:
    args = parse_arguments()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=args.timeout)
    except serial.SerialException as exc:
        print(f"Could not open {args.port}: {exc}")
        return 1

    data_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    reader = threading.Thread(target=serial_reader, args=(ser, data_queue, stop_event), daemon=True)
    reader.start()

    times = []
    channels = []
    lines = []
    start_time = None

    fig, ax = plt.subplots()
    ax.set_title("Live Serial Plot")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")

    def update(_frame: int) -> list[plt.Line2D]:
        nonlocal start_time
        updated = False

        while True:
            try:
                timestamp, values = data_queue.get_nowait()
            except queue.Empty:
                break

            if start_time is None:
                start_time = timestamp
            times.append(timestamp - start_time)

            if len(channels) < len(values):
                for _ in range(len(values) - len(channels)):
                    channels.append([float("nan")] * (len(times) - 1))
                    (line,) = ax.plot([], [], label=f"Channel {len(channels)}")
                    lines.append(line)
                ax.legend(loc="upper left")

            for index, channel in enumerate(channels):
                channel.append(values[index] if index < len(values) else float("nan"))

            if len(times) > args.window:
                cut = len(times) - args.window
                times[:] = times[cut:]
                for channel in channels:
                    channel[:] = channel[cut:]

            updated = True

        if updated:
            for line, channel in zip(lines, channels):
                line.set_data(times, channel)
            ax.relim()
            ax.autoscale_view()

        return lines

    animation = FuncAnimation(fig, update, interval=args.interval, blit=False)

    try:
        plt.show()
    finally:
        stop_event.set()
        reader.join(timeout=1.0)
        ser.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

