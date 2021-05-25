#pragma once

#include <SDL.h>
#include <SDL_thread.h>

#include "Packet.h"

class PacketQueue
{

public:
	PacketQueue();
	~PacketQueue();

	int put_private(AVPacket* pkt);
	int put(AVPacket* pkt);
	int put_nullpacket(int stream_index);
	void flush();
	void abort();
	void start();
	void destroy();
	int get(AVPacket* pkt, int block, int* serial);
	int init();

	Packet* first_pkt, * last_pkt;
	int nb_packets;
	int size;
	int64_t duration;
	int abort_request;
	int serial;
	SDL_mutex* mutex;
	SDL_cond* cond;

	AVPacket flush_pkt;
};

