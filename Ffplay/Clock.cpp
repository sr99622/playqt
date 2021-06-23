#include "Clock.h"
#include "mainwindow.h"

Clock::Clock()
{

}

double Clock::get_clock()
{
	if (*queue_serial != serial)
		return NAN;
	if (paused) {
		return pts;
	}
	else {
		double time = av_gettime_relative() / 1000000.0;
		return pts_drift + time - (time - last_updated) * (1.0 - speed);
	}
}

void Clock::set_clock_at(double pts, int serial, double time)
{
	this->pts = pts;
	this->last_updated = time;
	this->pts_drift = this->pts - time;
	this->serial = serial;
}

void Clock::set_clock(double pts, int serial)
{
	double time = av_gettime_relative() / 1000000.0;
	set_clock_at(pts, serial, time);
}

void Clock::set_clock_speed(double speed)
{
	set_clock(get_clock(), serial);
	this->speed = speed;
}

void Clock::init_clock(int* queue_serial)
{
	speed = 1.0;
	paused = 0;
	this->queue_serial = queue_serial;
	set_clock(NAN, -1);
}

void Clock::sync_clock_to_slave(Clock* slave)
{
	double clock = get_clock();
	double slave_clock = slave->get_clock();
	if (!isnan(slave_clock) && (isnan(clock) || fabs(clock - slave_clock) > AV_NOSYNC_THRESHOLD))
		set_clock(slave_clock, slave->serial);
}
