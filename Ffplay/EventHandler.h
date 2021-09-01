#pragma once

#include "VideoState.h"
#include <QMainWindow>

class EventHandler
{

public:
    EventHandler(QMainWindow *parent);
    void event_loop();
    bool looping = false;
    bool running = false;
    void feed();

    QMainWindow *mainWindow;
    SDL_Event event;

};

