/*******************************************************************************
* displaycontainer.h
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

#ifndef DISPLAYCONTAINER_H
#define DISPLAYCONTAINER_H

#include "displayslider.h"

#include <QObject>
#include <QWidget>
#include <QLabel>
#include <QSlider>
#include <QResizeEvent>
#include <QMainWindow>
#include <QMouseEvent>
#include <QLineEdit>
#include <QPushButton>

class DisplayLabel : public QLabel
{
    Q_OBJECT

public:
    DisplayLabel();
    QSize getSize();
    bool mouseTracking;

protected:
    void mouseReleaseEvent(QMouseEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void leaveEvent(QEvent *event) override;

signals:
    void clicked(QPoint);
    void rightClicked(QPoint);
    void mouseMoved(QMouseEvent*);
    void mouseLeft();

};

class DisplayContainer : public QFrame
{
    Q_OBJECT

public:
    DisplayContainer(QMainWindow *parent);
    void setIdle();
    void present(int current_stream_type);
    WId getWinId();
    void setDisplayHeight(int windowHeight);

    QMainWindow *mainWindow;
    DisplayLabel *display;
    DisplaySlider *slider;
    QLineEdit *elapsed;
    QLineEdit *total;
    QWidget *sliderPanel;

    QSize resolution;
    bool show_slider;

    const int displayInitialWidth = 960;
    const int displayInitialHeight = 540;

signals:
    //void sizeChanged(const QSize& size);

protected:
    void resizeEvent(QResizeEvent *event) override;

public slots:
    //void valueChanged(int arg);

};

#endif // DISPLAYCONTAINER_H
