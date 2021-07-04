#include "displayslider.h"
#include "mainwindow.h"

DisplaySlider::DisplaySlider(QMainWindow *parent) : QSlider(parent)
{
    mainWindow = parent;

    //setOrientation(Qt::Horizontal);
    //setMinimum(0);
    //setMaximum(1000);
    //setValue(0);

    //setMouseTracking(true);
    connect(this, SIGNAL(sliderMoved(int)), this, SLOT(moved(int)));
    connect(this, SIGNAL(sliderPressed()), this, SLOT(pressed()));
    connect(this, SIGNAL(sliderReleased()), this, SLOT(released()));
}

void DisplaySlider::mouseMoveEvent(QMouseEvent *e)
{
    //cout << "mouse position x: " << e->pos().x() << " y: " << e->pos().y() << endl;
}

void DisplaySlider::mousePressEvent(QMouseEvent *e)
{
    //QSlider::mousePressEvent(e);
}

void DisplaySlider::mouseReleaseEvent(QMouseEvent *e)
{
    //QSlider::mouseReleaseEvent(e);

    if (!MW->e->looping)
        return;

    tick = 1000 * e->position().x() / width();
    //if (MW->is)
    //    MW->is->stream_seek(tick, 0, 0);

    SDL_Event event;
    SDL_memset(&event, 0, sizeof(event));
    event.type = MW->sdlCustomEventType;
    event.user.code = SLIDER_POSITION_UPDATE;
    event.user.data1 = &tick;
    SDL_PushEvent(&event);

}

void DisplaySlider::keyPressEvent(QKeyEvent *event)
{
    //QSlider::keyPressEvent(event);
}

void DisplaySlider::moved(int value)
{

}

void DisplaySlider::pressed()
{

}

void DisplaySlider::released()
{

}
