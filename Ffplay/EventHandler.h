#pragma once

#include "VideoState.h"

#include <SDL.h>

class EventHandler
{

public:
	void event_loop(VideoState* cur_stream);
    bool running = false;

    double elapsed = 0;
    double total = 0;
    int percentage = 0;
    int64_t ts;

};

