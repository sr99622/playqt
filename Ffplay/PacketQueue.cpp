#include <iostream>

#include "PacketQueue.h"

using namespace std;

PacketQueue::PacketQueue()
{
	memset(this, 0, sizeof(PacketQueue));
	//mutex = SDL_CreateMutex();
	//cond = SDL_CreateCond();
	abort_request = 1;

	av_init_packet(&flush_pkt);
	flush_pkt.data = (uint8_t*)&flush_pkt;
}

PacketQueue::~PacketQueue()
{
	destroy();
}

int PacketQueue::init()
{
    mutex = SDL_CreateMutex();
    if (!mutex) {
        //av_log(NULL, AV_LOG_FATAL, "SDL_CreateMutex(): %s\n", SDL_GetError());
        cout << "SDL_CreateMutex error: " << SDL_GetError() << endl;
		return AVERROR(ENOMEM);
	}
    cond = SDL_CreateCond();
    if (!cond) {
        //av_log(NULL, AV_LOG_FATAL, "SDL_CreateCond(): %s\n", SDL_GetError());
        cout << "SDL_CreateCond error: " << SDL_GetError() << endl;
		return AVERROR(ENOMEM);
	}
    return 0;
}

int PacketQueue::get(AVPacket* pkt, int block, int* serial)
{
	Packet* pkt1;
	int ret;

	SDL_LockMutex(mutex);

	for (;;) {
		if (abort_request) {
			ret = -1;
			break;
		}

		pkt1 = first_pkt;
		if (pkt1) {
			first_pkt = pkt1->next;
			if (!first_pkt)
				last_pkt = NULL;
			nb_packets--;
			size -= pkt1->pkt.size + sizeof(*pkt1);
			duration -= pkt1->pkt.duration;
			*pkt = pkt1->pkt;
			if (serial)
				*serial = pkt1->serial;
			av_free(pkt1);
			ret = 1;
			break;
		}
		else if (!block) {
			ret = 0;
			break;
		}
		else {
			SDL_CondWait(cond, mutex);
		}
	}
	SDL_UnlockMutex(mutex);
    return ret;
}

int PacketQueue::put_private(AVPacket* pkt)
{
	Packet* pkt1;

	if (abort_request)
		return -1;

	//pkt1 = new Packet();
	pkt1 = (Packet*)av_malloc(sizeof(Packet));
	pkt1->pkt = *pkt;
	pkt1->next = NULL;

	if (pkt == &flush_pkt)
		serial++;
	pkt1->serial = serial;

	if (!last_pkt)
		first_pkt = pkt1;
	else
		last_pkt->next = pkt1;
	last_pkt = pkt1;
	nb_packets++;
	size += pkt1->pkt.size + sizeof(*pkt1);
	duration += pkt1->pkt.duration;
	SDL_CondSignal(cond);

	return 0;
}

int PacketQueue::put(AVPacket* pkt)
{
	int ret;

	SDL_LockMutex(mutex);
	ret = put_private(pkt);
	SDL_UnlockMutex(mutex);

	if (pkt != &flush_pkt && ret < 0)
		av_packet_unref(pkt);

	return ret;
}

int PacketQueue::put_nullpacket(int stream_index)
{
	AVPacket pkt1, * pkt = &pkt1;
	av_init_packet(pkt);
	pkt->data = NULL;
	pkt->size = 0;
	pkt->stream_index = stream_index;
	return put(pkt);
}

void PacketQueue::flush()
{
	Packet* pkt, * pkt1;

	SDL_LockMutex(mutex);
	for (pkt = first_pkt; pkt; pkt = pkt1) {
		pkt1 = pkt->next;
		av_packet_unref(&pkt->pkt);
		av_freep(&pkt);
	}
	last_pkt = NULL;
	first_pkt = NULL;
	nb_packets = 0;
	size = 0;
	duration = 0;
	SDL_UnlockMutex(mutex);
}

void PacketQueue::abort()
{
	SDL_LockMutex(mutex);
	abort_request = 1;
	SDL_CondSignal(cond);
	SDL_UnlockMutex(mutex);
}

void PacketQueue::start()
{
	SDL_LockMutex(mutex);
	abort_request = 0;
	put_private(&flush_pkt);
	SDL_UnlockMutex(mutex);
}

void PacketQueue::destroy()
{
	flush();
	SDL_DestroyMutex(mutex);
	SDL_DestroyCond(cond);
}
