/*******************************************************************************
* filterchain.h
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

#ifndef FILTERCHAIN_H
#define FILTERCHAIN_H

#include <QMainWindow>

#include "filter.h"
#include "filterpanel.h"
#include "Utilities/kalman.h"

#include <chrono>

using namespace std::chrono;

class FilterChain : public QObject
{
    Q_OBJECT


public:
    FilterChain(QMainWindow *parent);
    ~FilterChain();

    Frame *fp;
    Frame *vp;

    QMainWindow *mainWindow;
    Kalman k_time;
    Kalman k_fps;
    bool counting = false;
    int count = 0;
    high_resolution_clock::time_point t1;
    bool disengaged = false;

public slots:
    void process(Frame *vp);

};

#endif // FILTERCHAIN_H
