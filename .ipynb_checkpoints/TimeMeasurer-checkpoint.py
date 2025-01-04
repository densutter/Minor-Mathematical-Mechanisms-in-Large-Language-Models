import time


class TimeMeasurer:
    def __init__(self):
        # Initialize the start time (None initially)
        self.start_time = None
        self.total_time_estimate = None  # To store the estimated total time
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()  # Capture the current time in seconds
    
    def stop(self, progress: float):
        """
        Calculate the time left based on the progress (0 to 1) and the time elapsed so far.
        progress: A float indicating the completion percentage (0.0 to 1.0)
        """
        if self.start_time is None:
            raise ValueError("Timer has not been started yet. Please call start() before stop().")
        
        if not (0 <= progress <= 1):
            raise ValueError("Progress must be a float between 0 and 1.")

        # Capture the current time to measure elapsed time
        current_time = time.time()
        elapsed_time = current_time - self.start_time  # Time elapsed since the start

        # Estimate the total time based on the progress
        if progress > 0:
            total_time_estimate = elapsed_time / progress
        else:
            return "unknown"  # If no progress, assume no estimation

        # Calculate the remaining time
        time_left = total_time_estimate - elapsed_time

        # Convert the remaining time to days, hours, minutes, and seconds
        days = time_left // (24 * 3600)
        hours = (time_left % (24 * 3600)) // 3600
        minutes = (time_left % 3600) // 60
        seconds = time_left % 60

        # Return the formatted string
        return f"{int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"
