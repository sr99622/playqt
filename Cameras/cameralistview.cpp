/*******************************************************************************
* cameralistview.cpp
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

#include "cameralistview.h"
#include "mainwindow.h"
#include <QMouseEvent>

CameraListView::CameraListView(QMainWindow *parent)
{
    mainWindow = parent;
    cameraListModel = new CameraListModel(mainWindow);
    setModel(cameraListModel);
    setFrameStyle(QFrame::Panel | QFrame::Sunken);
    connect(selectionModel(), SIGNAL(selectionChanged(QItemSelection, QItemSelection)), cameraListModel, SLOT(onSelectedItemsChanged(QItemSelection, QItemSelection)));
}

void CameraListView::mouseDoubleClickEvent(QMouseEvent *event)
{
    //MW->cameraListDoubleClicked(currentIndex());
    Camera *camera = getCurrentCamera();
    if (camera) {
        QString rtsp = camera->onvif_data->stream_uri;
        QString username = camera->onvif_data->username;
        QString password = camera->onvif_data->password;
        QString str = rtsp.mid(0, 7) + username + ":" + password + "@" + rtsp.mid(7);
        MW->co->input_filename = av_strdup(str.toLatin1().data());
        MW->runLoop();
    }
}

Camera *CameraListView::getCurrentCamera()
{
    if (currentIndex().isValid())
        return (Camera*)((CameraListModel*)model())->cameras[currentIndex().row()];
    else
        return NULL;
}

void CameraListView::refresh()
{
    model()->emit dataChanged(QModelIndex(), QModelIndex());
}

void CameraListView::animateWriter()
{
    /*
    QPushButton *writerButton = MW->controlPanel->writerButton;
    if (writerButton->isEnabled()) {
        if (writerButton->isChecked())
            writerButton->animateClick(false);
        else
            writerButton->animateClick(true);
    }
    */
}

void CameraListView::animateSnapshot()
{
    /*
    QPushButton *snapshotButton = MW->controlPanel->snapshotButton;
    if (snapshotButton->isEnabled())
        snapshotButton->animateClick(true);
    */
}

void CameraListView::keyPressEvent(QKeyEvent *event)
{
    /*
    switch(event->key()) {
    case Qt::Key_Return:
        MW->cameraListDoubleClicked(currentIndex());
        setFocus();
        break;
    case Qt::Key_F4:
        animateWriter();
        break;
    case Qt::Key_F8:
        animateSnapshot();
        break;

    default:
        QListView::keyPressEvent(event);
    }
    */
}
