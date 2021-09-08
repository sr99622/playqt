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
    double elapsed = 0;
    double total = 0;
    int percentage = 0;
    int64_t ts;

};

