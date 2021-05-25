#pragma once

extern "C" {
#include "libavformat/avformat.h"
}

class Packet
{

public:
	Packet();

	AVPacket pkt;
	Packet* next;
	int serial;

};

