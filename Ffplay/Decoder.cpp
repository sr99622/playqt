#include "Decoder.h"

Decoder::Decoder()
{
	memset(this, 0, sizeof(Decoder));
	/*
	start_pts = AV_NOPTS_VALUE;
	next_pts = 0;
	pkt_serial = -1;
	finished = 0;
	packet_pending = 0;
	start_pts_tb = av_make_q(0, 1);
	next_pts_tb = av_make_q(0, 1);
	empty_queue_cond = NULL;
	decoder_tid = NULL;
	queue = NULL;
	avctx = NULL;
	*/
}

int Decoder::decode_frame(AVFrame* frame, AVSubtitle* sub)
{

	int ret = AVERROR(EAGAIN);

	for (;;) {
		AVPacket pkt1;

		if (queue->serial == pkt_serial) {
			do {
				if (queue->abort_request)
					return -1;

				switch (avctx->codec_type) {
				case AVMEDIA_TYPE_VIDEO:
					ret = avcodec_receive_frame(avctx, frame);
					if (ret >= 0) {
						if (decoder_reorder_pts == -1) {
							frame->pts = frame->best_effort_timestamp;
						}
						else if (!decoder_reorder_pts) {
							frame->pts = frame->pkt_dts;
						}
					}
					break;
				case AVMEDIA_TYPE_AUDIO:
					ret = avcodec_receive_frame(avctx, frame);
					if (ret >= 0) {
						AVRational tb = av_make_q(1, frame->sample_rate);
						if (frame->pts != AV_NOPTS_VALUE)
							frame->pts = av_rescale_q(frame->pts, avctx->pkt_timebase, tb);
						else if (next_pts != AV_NOPTS_VALUE)
							frame->pts = av_rescale_q(next_pts, next_pts_tb, tb);
						if (frame->pts != AV_NOPTS_VALUE) {
							next_pts = frame->pts + frame->nb_samples;
							next_pts_tb = tb;
						}
					}
					break;
				}
				if (ret == AVERROR_EOF) {
					finished = pkt_serial;
					avcodec_flush_buffers(avctx);
					return 0;
				}
				if (ret >= 0)
					return 1;
			} while (ret != AVERROR(EAGAIN));
		}

		do {
			if (queue->nb_packets == 0)
				SDL_CondSignal(empty_queue_cond);
			if (packet_pending) {
				av_packet_move_ref(&pkt1, &pkt);
				packet_pending = 0;
			}
			else {
				if (queue->get(&pkt1, 1, &pkt_serial) < 0)
					return -1;
			}
		} while (queue->serial != pkt_serial);

		if (pkt1.data == flush_pkt.data) {
			avcodec_flush_buffers(avctx);
			finished = 0;
			next_pts = start_pts;
			next_pts_tb = start_pts_tb;
		}
		else {
			if (avctx->codec_type == AVMEDIA_TYPE_SUBTITLE) {
				int got_frame = 0;
				ret = avcodec_decode_subtitle2(avctx, sub, &got_frame, &pkt1);
				if (ret < 0) {
					ret = AVERROR(EAGAIN);
				}
				else {
					if (got_frame && !pkt1.data) {
						packet_pending = 1;
						av_packet_move_ref(&pkt, &pkt1);
					}
					ret = got_frame ? 0 : (pkt1.data ? AVERROR(EAGAIN) : AVERROR_EOF);
				}
			}
			else {
				if (avcodec_send_packet(avctx, &pkt1) == AVERROR(EAGAIN)) {
					av_log(avctx, AV_LOG_ERROR, "Receive_frame and send_packet both returned EAGAIN, which is an API violation.\n");
					packet_pending = 1;
					av_packet_move_ref(&pkt, &pkt1);
				}
			}
			av_packet_unref(&pkt1);
		}
	}
}

int Decoder::start(int (*fn)(void*), const char* thread_name, void* arg)
{
	queue->start();
	decoder_tid = SDL_CreateThread(fn, thread_name, arg);
	if (!decoder_tid) {
		return AVERROR(ENOMEM);
	}
	return 0;
}

void Decoder::init(AVCodecContext* avctx, PacketQueue* queue, SDL_cond* empty_queue_cond, AVPacket* flush_pkt)
{
	this->avctx = avctx;
	this->queue = queue;
	this->empty_queue_cond = empty_queue_cond;
	start_pts = AV_NOPTS_VALUE;
	pkt_serial = -1;
	this->flush_pkt = *flush_pkt;
}

void Decoder::abort(FrameQueue* fq)
{
	queue->abort();
	fq->signal();
	SDL_WaitThread(decoder_tid, NULL);
	decoder_tid = NULL;
	queue->flush();
}

void Decoder::destroy()
{
	av_packet_unref(&pkt);
	avcodec_free_context(&avctx);
}
