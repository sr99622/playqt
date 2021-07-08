/*******************************************************************************
* videotab.h
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

#ifndef VIDEOTAB_H
#define VIDEOTAB_H

#include "cameradialogtab.h"
#include "onvifmanager.h"

#include <QComboBox>
#include <QSpinBox>

class VideoTab : public CameraDialogTab
{
    Q_OBJECT

public:
    VideoTab(QWidget *parent);

    QComboBox *comboResolutions;
    QSpinBox *spinFrameRate;
    QSpinBox *spinGovLength;
    QSpinBox *spinBitrate;

    QWidget *cameraPanel;

    VideoUpdater *updater;

    void update() override;
    void setActive(bool active) override;
    bool hasBeenEdited() override;

public slots:
    void initialize();

private slots:
    void onCurrentIndexChanged(int index);
    void onValueChanged(int value);
};

#endif // VIDEOTAB_H
