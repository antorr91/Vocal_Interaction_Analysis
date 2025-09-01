import math

def validate_offset(reference_event, estimated_event, t_collar=0.200, percentage_of_length=0.5):
        """Validate estimated event based on event offset

        Parameters
        ----------
        reference_event : dict
            Reference event.

        estimated_event : dict
            Estimated event.

        t_collar : float > 0, seconds
            First condition, Time collar with which the estimated offset has to be in order to be consider valid estimation.
            Default value 0.2

        percentage_of_length : float in [0, 1]
            Second condition, percentage of the length within which the estimated offset has to be in order to be
            consider valid estimation.
            Default value 0.5

        Returns
        -------
        bool

        """

        # Detect field naming style used and validate onset
        if 'event_offset' in reference_event and 'event_offset' in estimated_event:
            annotated_length = reference_event['event_offset'] - reference_event['event_onset']

            return math.fabs(reference_event['event_offset'] - estimated_event['event_offset']) <= max(t_collar, percentage_of_length * annotated_length)

        elif 'offset' in reference_event and 'offset' in estimated_event:
            annotated_length = reference_event['offset'] - reference_event['onset']

            return math.fabs(reference_event['offset'] - estimated_event['offset']) <= max(t_collar, percentage_of_length * annotated_length)

