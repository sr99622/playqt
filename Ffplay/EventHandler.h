#pragma once

#include "VideoState.h"
#include <QMainWindow>

#include <SDL.h>

class EventHandler : public QObject
{
    Q_OBJECT

public:
    EventHandler(QMainWindow *parent);
    void event_loop();
    bool running = false;

    QMainWindow *mainWindow;

    SDL_Event event;
    double elapsed = 0;
    double total = 0;
    double remaining_time;
    int percentage = 0;
    int64_t ts;

public slots:
    void feed();
    void guiUpdate(int);

};

