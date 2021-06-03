#pragma once

#include "PacketQueue.h"
#include "FrameQueue.h"

class Decoder
{

public:
	Decoder();
	void init(AVCodecContext* avctx, PacketQueue* queue, SDL_cond* empty_queue_cond, AVPacket* flush_pkt);
	int start(int (*fn)(void*), const char* thread_name, void* arg);
	void abort(FrameQueue* fq);
	void destroy();
	int decode_frame(AVFrame* frame, AVSubtitle* sub);

	int decoder_reorder_pts = -1;
	AVPacket flush_pkt;

	AVPacket pkt;
	PacketQueue* queue;
	AVCodecContext* avctx;
	int pkt_serial;
	int finished;
	int packet_pending;
	int64_t start_pts;
	int64_t next_pts;
	AVRational start_pts_tb;
	AVRational next_pts_tb;
	SDL_cond* empty_queue_cond;
	SDL_Thread* decoder_tid;
};

