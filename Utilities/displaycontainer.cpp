/*******************************************************************************
* displaycontainer.cpp
*
* Copyright (c) 2020 Stephen Rhodes
*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along
* with this program; if not, write to the Free Software Foundation, Inc.,
* 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*
*******************************************************************************/

#include "displaycontainer.h"
#include "mainwindow.h"

#include <QSize>
#include <QGridLayout>

DisplayContainer::DisplayContainer(QMainWindow *parent)
{
    mainWindow = parent;
    display = new DisplayLabel();
    display->setMinimumWidth(displayInitialWidth);
    display->setMinimumHeight(displayInitialHeight);
    //display->setUpdatesEnabled(false);  // This setting prevents flicker in the display, deferred to render
    QGridLayout *layout = new QGridLayout();
    slider = new DisplaySlider(mainWindow);
    //slider->setOrientation(Qt::Horizontal);
    //slider->setMinimum(0);
    //slider->setMaximum(1000);
    //slider->setValue(0);

    elapsed = new QLineEdit;
    elapsed->setMaximumWidth(elapsed->fontMetrics().boundingRect("00:00:00").width() * 1.2);
    elapsed->setAlignment(Qt::AlignHCenter);
    elapsed->setReadOnly(true);

    total = new QLineEdit;
    total->setMaximumWidth(total->fontMetrics().boundingRect("00:00:00").width() * 1.2);
    total->setAlignment(Qt::AlignHCenter);
    total->setReadOnly(true);

    sliderPanel = new QWidget;
    QHBoxLayout *sliderLayout = new QHBoxLayout;
    sliderLayout->addWidget(elapsed);
    sliderLayout->addWidget(slider);
    sliderLayout->addWidget(total);
    sliderPanel->setLayout(sliderLayout);
    sliderPanel->setMaximumHeight(elapsed->fontMetrics().height() * 3.0);

    layout->addWidget(display, 0, 0, 1, 2);
    layout->addWidget(sliderPanel, 1, 0, 1, 2);
    layout->setSpacing(0);

    setLayout(layout);
    //setFrameStyle(QFrame::Panel | QFrame::Sunken);
    show_slider = true;

}

DisplayLabel::DisplayLabel() : QLabel()
{
    mouseTracking = false;
}

QSize DisplayLabel::getSize()
{
    return size();
}

void DisplayLabel::leaveEvent(QEvent *event)
{
    if (mouseTracking) {
        emit mouseLeft();
    }
}

void DisplayLabel::mouseMoveEvent(QMouseEvent *event)
{
    if (mouseTracking) {
        //cout << "mouseMoved: " << event->pos().x() << ", " << event->pos().y() << endl;
        emit mouseMoved(event);
    }
    //QLabel::mouseMoveEvent(event);
}

void DisplayLabel::mouseReleaseEvent(QMouseEvent *event)
{
    if (!mouseTracking) {
        QLabel::mouseReleaseEvent(event);
        if (event->button() == Qt::RightButton)
           emit rightClicked(event->pos());
        else
            emit clicked(event->pos());
    }
    else {
        emit mouseMoved(event);
    }
}

void DisplayLabel::mousePressEvent(QMouseEvent *event)
{
    if (mouseTracking) {
        emit mouseMoved(event);
    }
}

void DisplayContainer::resizeEvent(QResizeEvent *event)
{
    cout << "DisplayContainer width: " << event->size().width() << " height: " << event->size().height() << endl;
    //if (event->size().width() == MW->splitter->sizes()[splitter_index])
    //    emit sizeChanged(event->size());
}

WId DisplayContainer::getWinId()
{
    return display->winId();
}

void DisplayContainer::setIdle()
{
    display->hide();
    slider->setValue(0);
    sliderPanel->hide();
}

void DisplayContainer::setDisplayHeight(int windowHeight)
{
    int displayHeight = windowHeight - windowHeightDifferential;
    //display->setMinimumHeight(displayHeight);
    //display->setMaximumHeight(displayHeight);
}

void DisplayContainer::present(int current_stream_type)
{
    if (show_slider) {
        //if (current_stream_type == VIDEO_STREAM) {
            sliderPanel->show();
        //}
        //else {
        //    slider->setValue(0);
        //    sliderPanel->hide();
        //}
    }
    else {
        slider->setValue(0);
        sliderPanel->hide();
    }
    display->show();
}
