#include <algorithm>

#include "FrameQueue.h"

using namespace std;

FrameQueue::FrameQueue()
{
	memset(this, 0, sizeof(FrameQueue));
	/*
	rindex = 0;
	windex = 0;
	size = 0;
	max_size = 0;
	keep_last = 0;
	rindex_shown = 0;
    pktq = NULL;
	mutex = SDL_CreateMutex();
	cond = SDL_CreateCond();
	*/
}

int FrameQueue::init(PacketQueue* pktq, int max_size, int keep_last)
{
	if (!(mutex = SDL_CreateMutex())) {
		av_log(NULL, AV_LOG_FATAL, "SDL_CreateMutex(): %s\n", SDL_GetError());
		return AVERROR(ENOMEM);
	}
	if (!(cond = SDL_CreateCond())) {
		av_log(NULL, AV_LOG_FATAL, "SDL_CreateCond(): %s\n", SDL_GetError());
		return AVERROR(ENOMEM);
	}

    this->pktq = pktq;
    this->max_size = min(max_size, FRAME_QUEUE_SIZE);
    this->keep_last = !!keep_last;
    for (int i = 0; i < this->max_size; i++)
        if (!(queue[i].frame = av_frame_alloc()))
            return AVERROR(ENOMEM);

    return 0;
}

void FrameQueue::unref_item(Frame* f)
{
	av_frame_unref(f->frame);
	avsubtitle_free(&f->sub);
}

void FrameQueue::destroy()
{
	for (int i = 0; i < max_size; i++) {
		unref_item(&queue[i]);
		av_frame_free(&queue[i].frame);
	}
	SDL_DestroyMutex(mutex);
	SDL_DestroyCond(cond);
}

void FrameQueue::signal()
{
	SDL_LockMutex(mutex);
	SDL_CondSignal(cond);
	SDL_UnlockMutex(mutex);
}

void FrameQueue::push()
{
	if (++windex == max_size)
		windex = 0;
	SDL_LockMutex(mutex);
	size++;
	SDL_CondSignal(cond);
	SDL_UnlockMutex(mutex);
}

void FrameQueue::next()
{
	if (keep_last && !rindex_shown) {
		rindex_shown = 1;
		return;
	}
	unref_item(&queue[rindex]);
	if (++rindex == max_size)
		rindex = 0;
	SDL_LockMutex(mutex);
	size--;
	SDL_CondSignal(cond);
	SDL_UnlockMutex(mutex);
}

int FrameQueue::nb_remaining()
{
	return size - rindex_shown;
}

int64_t FrameQueue::last_pos()
{
	Frame* fp = &queue[rindex];
	if (rindex_shown && fp->serial == pktq->serial)
		return fp->pos;
	else
		return -1;
}

Frame* FrameQueue::peek()
{
	return &queue[(rindex + rindex_shown) % max_size];
}

Frame* FrameQueue::peek_next()
{
	return &queue[(rindex + rindex_shown + 1) % max_size];
}

Frame* FrameQueue::peek_last()
{
	return &queue[rindex];
}

Frame* FrameQueue::peek_writable()
{
	SDL_LockMutex(mutex);
    while (size >= max_size && !pktq->abort_request) {
        SDL_CondWait(cond, mutex);
    }
    SDL_UnlockMutex(mutex);

	if (pktq->abort_request)
		return NULL;

    return &queue[windex];
}

Frame* FrameQueue::peek_readable()
{
	SDL_LockMutex(mutex);
    while (size - rindex_shown <= 0 && !pktq->abort_request) {
		SDL_CondWait(cond, mutex);
	}
	SDL_UnlockMutex(mutex);

	if (pktq->abort_request)
		return NULL;

	return &queue[(rindex + rindex_shown) % max_size];
}
