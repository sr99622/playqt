#pragma once

#include "PacketQueue.h"
#include "Frame.h"

#define FRAME_QUEUE_SIZE 16

class FrameQueue
{
public:
	FrameQueue();
	int init(PacketQueue* pktq, int max_size, int keep_last);
	void unref_item(Frame* f);
	void destroy();
	void signal();
	void push();
	void next();
	Frame* peek();
	Frame* peek_next();
	Frame* peek_last();
	Frame* peek_writable();
	Frame* peek_readable();
	int nb_remaining();
	int64_t last_pos();

	Frame queue[FRAME_QUEUE_SIZE];
	int rindex;
	int windex;
	int size;
	int max_size;
	int keep_last;
	int rindex_shown;
	SDL_mutex* mutex;
	SDL_cond* cond;
	PacketQueue* pktq;
};

