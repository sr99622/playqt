#pragma once

#include "VideoState.h"

#include <SDL.h>

class EventHandler
{
public:
	void event_loop(VideoState* cur_stream);
};

