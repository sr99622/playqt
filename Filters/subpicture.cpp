/*******************************************************************************
* subpicture.cpp
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

#include "subpicture.h"
#include "mainwindow.h"

#include <QGridLayout>
#include <QLabel>
#include <QTextStream>

SubPicture::SubPicture(QMainWindow *parent)
{
    name = "Sub Picture";
    panel = new Panel(mainWindow);
    mainWindow = parent;

    textX = new NumberTextBox;
    textY = new NumberTextBox;
    textW = new NumberTextBox;
    textH = new NumberTextBox;

    int textWidth = textX->fontMetrics().boundingRect("00000").width() * 1.5;
    textX->setMaximumWidth(textWidth);
    textY->setMaximumWidth(textWidth);
    textW->setMaximumWidth(textWidth);
    textH->setMaximumWidth(textWidth);

    buttonApply = new QPushButton("...");
    buttonApply->setMaximumWidth(50);
    connect(buttonApply, SIGNAL(clicked()), this, SLOT(apply()));

    QWidget *coordinatePanel = new QWidget;
    QGridLayout *coordinateLayout = new QGridLayout;
    coordinateLayout->addWidget(new QLabel("x"),   0, 0, 1, 1, Qt::AlignRight);
    coordinateLayout->addWidget(textX,             0, 1, 1, 1);
    coordinateLayout->addWidget(new QLabel("y"),   0, 2, 1, 1, Qt::AlignRight);
    coordinateLayout->addWidget(textY,             0, 3, 1, 1);
    coordinateLayout->addWidget(new QLabel("w"),   0, 4, 1, 1, Qt::AlignRight);
    coordinateLayout->addWidget(textW,             0, 5, 1, 1);
    coordinateLayout->addWidget(new QLabel("h"),   0, 6, 1, 1, Qt::AlignRight);
    coordinateLayout->addWidget(textH,             0, 7, 1, 1);
    coordinateLayout->addWidget(buttonApply,       0, 8, 1, 1);
    coordinatePanel->setLayout(coordinateLayout);

    buttonUp = new QPushButton("^");
    buttonDown = new QPushButton("v");
    buttonLeft = new QPushButton("<");
    buttonRight = new QPushButton(">");

    buttonUp->setMaximumWidth(50);
    buttonDown->setMaximumWidth(50);
    buttonLeft->setMaximumWidth(50);
    buttonRight->setMaximumWidth(50);

    connect(buttonUp, &QPushButton::pressed, [=] {move(0, 1, 0);});
    connect(buttonDown, &QPushButton::pressed, [=] {move(0, -1, 0);});
    connect(buttonLeft, &QPushButton::pressed, [=] {move(-1, 0, 0);});
    connect(buttonRight, &QPushButton::pressed, [=] {move(1, 0, 0);});
    connect(buttonUp, SIGNAL(released()), this, SLOT(stop()));
    connect(buttonDown, SIGNAL(released()), this, SLOT(stop()));
    connect(buttonLeft, SIGNAL(released()), this, SLOT(stop()));
    connect(buttonRight, SIGNAL(released()), this, SLOT(stop()));

    QWidget *directionPanel = new QWidget;
    QGridLayout *directionLayout = new QGridLayout;
    directionLayout->addWidget(buttonUp,        0, 1, 1, 1);
    directionLayout->addWidget(buttonLeft,      1, 0, 1, 1);
    directionLayout->addWidget(buttonRight,     1, 2, 1, 1);
    directionLayout->addWidget(buttonDown,      2, 1, 1, 1);
    directionLayout->setContentsMargins(11, 0, 0, 0);
    directionPanel->setLayout(directionLayout);

    buttonZoomIn = new QPushButton("Zoom In");
    buttonZoomOut = new QPushButton("Zoom Out");
    int zoomWidth = buttonZoomOut->fontMetrics().boundingRect("Zoom Out").width() * 1.5;
    buttonZoomIn->setMaximumWidth(zoomWidth);
    buttonZoomOut->setMaximumWidth(zoomWidth);
    connect(buttonZoomIn, &QPushButton::pressed, [=] {move(0, 0, 1);});
    connect(buttonZoomOut, &QPushButton::pressed, [=] {move(0, 0, -1);});
    connect(buttonZoomIn, SIGNAL(released()), this, SLOT(stop()));
    connect(buttonZoomOut, SIGNAL(released()), this, SLOT(stop()));

    QWidget *zoomPanel = new QWidget;
    QGridLayout *zoomLayout = new QGridLayout;
    zoomLayout->addWidget(buttonZoomIn,    0, 0, 1, 1);
    zoomLayout->addWidget(buttonZoomOut,   1, 0, 1, 1);
    zoomLayout->setContentsMargins(0, 0, 0, 0);
    zoomPanel->setLayout(zoomLayout);

    checkPreset = new QCheckBox("Set PTZ Position");

    button1 = new QPushButton("1");
    button2 = new QPushButton("2");
    button3 = new QPushButton("3");
    button4 = new QPushButton("4");
    button5 = new QPushButton("5");

    button1->setMinimumWidth(40);
    button2->setMinimumWidth(40);
    button3->setMinimumWidth(40);
    button4->setMinimumWidth(40);
    button5->setMinimumWidth(40);

    connect(button1, &QPushButton::clicked, [=] {preset(0);});
    connect(button2, &QPushButton::clicked, [=] {preset(1);});
    connect(button3, &QPushButton::clicked, [=] {preset(2);});
    connect(button4, &QPushButton::clicked, [=] {preset(3);});

    autoLoad = new QCheckBox("Auto Load Preset ");
    autoLoad->setChecked(MW->settings->value(autoLoadKey, false).toBool());
    connect(autoLoad, SIGNAL(clicked(bool)), this, SLOT(autoLoadClicked(bool)));
    QStringList list;
    list << "1" << "2" << "3" << "4" << "5";
    autoPreset = new QComboBox();
    autoPreset->addItems(list);
    autoPreset->setMaximumWidth(50);
    autoPreset->setCurrentIndex(MW->settings->value(autoPresetKey, 0).toInt());
    connect(autoPreset, SIGNAL(currentIndexChanged(int)), this, SLOT(autoPresetChanged(int)));
    QLabel *lbl00 = new QLabel(" on Startup");

    buttonReset = new QPushButton("Reset");
    connect(buttonReset, SIGNAL(clicked()), this, SLOT(reset()));
    connect(button5, &QPushButton::clicked, [=] {preset(4);});

    QWidget *footPanel = new QWidget();
    QGridLayout *footLayout = new QGridLayout();
    footLayout->addWidget(autoLoad,    0, 0, 1, 1);
    footLayout->addWidget(autoPreset,  0, 1, 1, 1);
    footLayout->addWidget(lbl00,       0, 2, 1, 1);
    footLayout->addWidget(buttonReset, 0, 3, 1, 1);
    footPanel->setLayout(footLayout);

    QGridLayout *layout = new QGridLayout;
    layout->addWidget(coordinatePanel, 1, 0, 1, 10);
    layout->addWidget(directionPanel,  2, 0, 1, 6);
    layout->addWidget(zoomPanel,       2, 6, 1, 4);
    layout->addWidget(checkPreset,     3, 5, 1, 5, Qt::AlignCenter);
    layout->addWidget(button1,         4, 0, 1, 2, Qt::AlignCenter);
    layout->addWidget(button2,         4, 2, 1, 2, Qt::AlignCenter);
    layout->addWidget(button3,         4, 4, 1, 2, Qt::AlignCenter);
    layout->addWidget(button4,         4, 6, 1, 2, Qt::AlignCenter);
    layout->addWidget(button5,         4, 8, 1, 2, Qt::AlignCenter);
    layout->addWidget(footPanel,       5, 0, 1, 10);
    layout->setContentsMargins(0, 0, 0, 0);
    panel->setLayout(layout);

    for (int i = 0; i < 5; i++) {
        if (MW->settings->contains(presetKey + QString::number(i))) {
            presets[i] = MW->settings->value(presetKey + QString::number(i)).toRect();
        }
    }

    moving = false;
    scale = 1.0;
    denominator = ZOOM_FACTOR;
}

void SubPicture::filter(Frame *vp)
{
    if (codec_width != MW->is->video_st->codecpar->width || codec_height != MW->is->video_st->codecpar->height)
        reset();

    if (first_pass) {
        if (autoLoad->isChecked()) {
            QRect rect = presets[autoPreset->currentIndex()];
            bool wider = rect.x() + rect.width() > codec_width;
            bool higher = rect.y() + rect.height() > codec_height;
            if (!wider && !higher) {
                update(rect.x(), rect.y(), rect.width(), rect.height());
                denominator = ZOOM_FACTOR * codec_width / (float) w;
            }
        }
        first_pass = false;
        return;
    }

    ptz();

    Frame dummy(w, h, (AVPixelFormat)vp->format);

    if (vp->width == codec_width && vp->height == codec_height) {
        vp->slice(x, y, &dummy);
        vp->allocateFrame(dummy.width, dummy.height, (AVPixelFormat)dummy.format);
        av_frame_copy(vp->frame, dummy.frame);
    }
    else {
        if (MW->is->paused)
            MW->is->step_to_next_frame();
    }

}

void SubPicture::autoLoadClicked(bool checked)
{
    MW->settings->setValue(autoLoadKey, checked);
}

void SubPicture::autoPresetChanged(int index)
{
    MW->settings->setValue(autoPresetKey, index);
}

void SubPicture::apply()
{
    update(textX->text().toInt(), textY->text().toInt(), textW->text().toInt(), textH->text().toInt());
    denominator = ZOOM_FACTOR * codec_width / (float) w;
}

void SubPicture::keyReleaseEvent(QKeyEvent *event)
{
    if (event->isAutoRepeat())
        return;

    if (event->modifiers() & Qt::ControlModifier) {
        switch (event->key()) {
        case Qt::Key_Semicolon:
            stop();
            break;
        case Qt::Key_Apostrophe:
            stop();
            break;
        case Qt::Key_BracketLeft:
            stop();
            break;
        case Qt::Key_Slash:
            stop();
            break;
        case Qt::Key_Comma:
            stop();
            break;
        case Qt::Key_Period:
            stop();
            break;
        }
    }
}

void SubPicture::keyPressEvent(QKeyEvent *event)
{

    if (event->isAutoRepeat())
        return;

    if (event->modifiers() & Qt::ControlModifier) {
        switch (event->key()) {
        case Qt::Key_Semicolon:
            move(-1, 0, 0);
            break;
        case Qt::Key_Apostrophe:
            move(1, 0, 0);
            break;
        case Qt::Key_BracketLeft:
            move(0, 1, 0);
            break;
        case Qt::Key_Slash:
            move(0, -1, 0);
            break;
        case Qt::Key_Comma:
            move(0, 0, 1);
            break;
        case Qt::Key_Period:
            move(0, 0, -1);
            break;
        }
    }
}

void SubPicture::update(int x_arg, int y_arg, int w_arg, int h_arg)
{
    x = x_arg; y = y_arg; w = w_arg; h = h_arg;

    w = w + w%2;
    h = h + h%2;
    x = x + x%2;
    y = y + y%2;

    if ((x + w) > codec_width - 2)
        x = codec_width - w - 2;

    if ((y + h) > codec_height - 2)
        y = codec_height - h - 2;

    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (w < 2) w = 2;
    if (h < 2) h = 2;

    textX->setText(QString::number(x));
    textY->setText(QString::number(y));
    textW->setText(QString::number(w));
    textH->setText(QString::number(h));
}

void SubPicture::reset()
{
    if (MW->is != nullptr) {
        codec_width = MW->is->video_st->codecpar->width;
        codec_height = MW->is->video_st->codecpar->height;
    }
    else {
        codec_width = 0;
        codec_height = 0;
    }

    update(0, 0, codec_width, codec_height);
    denominator = ZOOM_FACTOR;
}

void SubPicture::move(int p, int t, int z)
{
    pan = p;
    tilt = t;
    zoom = z;
    moving = true;
    ptz();
}

void SubPicture::stop()
{
    moving = false;
}

void SubPicture::preset(int arg) {
    if (checkPreset->isChecked()) {
        QRect rect(x, y, w, h);
        presets[arg] = rect;
        MW->settings->setValue(presetKey + QString::number(arg), rect);
        checkPreset->setChecked(false);
    }
    else {
        if (presets[arg].x() == 0 && presets[arg].y() == 0 && presets[arg].width() == 0 && presets[arg].height() == 0) {
            MW->msg(QString("No value found for preset %1").arg(arg+1));
        }
        else if (presets[arg].x() + presets[arg].width() > codec_width || presets[arg].y() + presets[arg].height() > codec_height) {
            MW->msg(QString("Invalid preset values for this resolution"));
        }
        else {
            update(presets[arg].x(), presets[arg].y(), presets[arg].width(), presets[arg].height());
            denominator = ZOOM_FACTOR * codec_width / (float) w;
        }
    }
}

void SubPicture::ptz()
{
    if (moving) {
        int ptz_x = x;
        int ptz_y = y;
        int ptz_w = w;
        int ptz_h = h;

        if (zoom > 0) {
            denominator++;
            scale = ZOOM_FACTOR / (float) denominator;

            int old_w = ptz_w;
            int old_h = ptz_h;

            ptz_w = codec_width * scale;
            ptz_h = codec_height * scale;

            ptz_x = (2*ptz_x + old_w - ptz_w) / 2;
            ptz_y = (2*ptz_y + old_h - ptz_h) / 2;
        }
        else if (zoom < 0) {
            denominator--;
            if (denominator < ZOOM_FACTOR)
                denominator = ZOOM_FACTOR;
            scale = ZOOM_FACTOR / (float) denominator;

            int old_w = ptz_w;
            int old_h = ptz_h;

            ptz_w = codec_width * scale;
            ptz_h = codec_height * scale;

            ptz_x = (2*ptz_x + old_w - ptz_w) / 2;
            ptz_y = (2*ptz_y + old_h - ptz_h) / 2;
        }
        else if (pan > 0) {
            ptz_x += 10;
        }
        else if (pan < 0) {
            ptz_x -= 10;
        }
        else if (tilt < 0) {
            ptz_y += 10;
        }
        else if (tilt > 0) {
            ptz_y -= 10;
        }

        update(ptz_x, ptz_y, ptz_w, ptz_h);
    }
}
