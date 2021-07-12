/*******************************************************************************
* filter.h
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

#ifndef FILTER_H
#define FILTER_H

#include <QObject>
#include <QWidget>
#include <QSettings>

#include "Ffplay/Frame.h"

class Filter : public QObject
{
    Q_OBJECT

public:
    Filter();

    virtual void filter(Frame *vp);
    virtual void initialize();
    virtual void keyPressEvent(QKeyEvent *event);
    virtual void keyReleaseEvent(QKeyEvent *event);
    virtual void deactivate();
    virtual void saveSettings(QSettings *settings);
    virtual void restoreSettings(QSettings *settings);

    QString name;
    QWidget *panel;
};

#endif // FILTER_H
