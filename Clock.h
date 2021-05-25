#pragma once

extern "C" {
#include "libavutil/time.h"
#include "libavutil/mathematics.h"
}

#define AV_NOSYNC_THRESHOLD 10.0

class Clock
{

public:
	Clock();
	double get_clock();
	void set_clock_at(double pts, int serial, double time);
	void set_clock(double pts, int serial);
	void set_clock_speed(double speed);
	void init_clock(int* queue_serial);
	void sync_clock_to_slave(Clock* slave);

	double pts;
	double pts_drift;
	double last_updated;
	double speed;
	int serial;
	int paused;
	int* queue_serial;
};

