/*******************************************************************************
* subpicture.h
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

#ifndef FILTERSUBPICTURE_H
#define FILTERSUBPICTURE_H

#include "filter.h"
#include "Utilities/numbertextbox.h"
#include <QPushButton>
#include <QCheckBox>
#include <QRect>
#include <QObject>
#include <QMainWindow>

#define ZOOM_FACTOR 8

class SubPicture : public Filter
{
    Q_OBJECT

public:
    SubPicture(QMainWindow *parent);

    void filter(Frame *vp) override;
    void initialize() override;
    void saveSettings(QSettings *settings) override;
    void restoreSettings(QSettings *settings) override;
    void ptz();

    Frame sub_frame;

    QMainWindow *mainWindow;

    QPushButton *button1;
    QPushButton *button2;
    QPushButton *button3;
    QPushButton *button4;
    QPushButton *button5;

    QPushButton *buttonUp;
    QPushButton *buttonDown;
    QPushButton *buttonLeft;
    QPushButton *buttonRight;
    QPushButton *buttonZoomIn;
    QPushButton *buttonZoomOut;

    QPushButton *buttonReset;
    QPushButton *buttonApply;

    NumberTextBox *textX;
    NumberTextBox *textY;
    NumberTextBox *textW;
    NumberTextBox *textH;

    int x = 0, y = 0, w = 0, h = 0;
    int pan, tilt, zoom;

    bool moving;
    float scale;
    int denominator;

    QRect presets[5] = {QRect(0,0,0,0)};
    QCheckBox *checkPreset;

    int codec_width = 0, codec_height = 0;

    const char * buttonStyle =  "QPushButton{                 \
                                background-color:lightGrey;   \
                                }";

signals:
    void msg(QString);

public slots:
    void move(int p, int t, int z);
    void stop();
    void apply();
    void update(int x_arg, int y_arg, int w_arg, int h_arg);
    void reset();
    void preset(int arg);

};

#endif // FILTERSUBPICTURE_H
