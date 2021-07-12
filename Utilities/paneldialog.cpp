/*******************************************************************************
* panledialog.cpp
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

#include "paneldialog.h"
#include "mainwindow.h"

PanelDialog::PanelDialog(QMainWindow *parent) : QDialog(parent, Qt::WindowSystemMenuHint | Qt::WindowTitleHint | Qt::WindowCloseButtonHint)
{
    mainWindow = parent;
}

void PanelDialog::keyPressEvent(QKeyEvent *event)
{
    if (event->modifiers() & Qt::ControlModifier) {
        switch (event->key()) {
        case Qt::Key_O:
            break;
        case Qt::Key_X:
            MW->close();
            break;
        case Qt::Key_P:
            MW->mainPanel->controlPanel->play();
            break;
        case Qt::Key_R:
            MW->mainPanel->controlPanel->rewind();
            break;
        case Qt::Key_T:
            MW->mainPanel->controlPanel->fastforward();
            break;
        case Qt::Key_V:
            MW->mainPanel->controlPanel->previous();
            break;
        case Qt::Key_N:
            MW->mainPanel->controlPanel->next();
            break;
        case Qt::Key_M:
            MW->mainPanel->controlPanel->mute();
            break;
        case Qt::Key_Q:
            MW->mainPanel->controlPanel->quit();
            break;
        case Qt::Key_E:
            MW->filterDialog->panel->engageFilter->setChecked(!MW->filterDialog->panel->engageFilter->isChecked());
            break;
        case Qt::Key_S:
            MW->parameterDialog->show();
            break;
        }
    }

    /*
    if (isCtrl && event->key() == Qt::Key_P)
        MW->mainPanel->controlPanel->play();
    */

    QDialog::keyPressEvent(event);
    //MW->getKeyEvent(event);
}

void PanelDialog::showEvent(QShowEvent *event)
{
    shown = true;

    if (gm.width() == 0) {

        int w = getDefaultWidth();
        int h = getDefaultHeight();
        if (getSettingsKey().length() > 0) {
            if (MW->settings->contains(getSettingsKey())) {
                QSize size = MW->settings->value(getSettingsKey()).toSize();
                w = size.width();
                h = size.height();
            }
        }

        int x = MW->geometry().center().x() - w/2;
        int y = MW->geometry().center().y() - h/2;
        gm = QRect(x, y, w, h);
    }

    setGeometry(gm);
    QDialog::showEvent(event);
}

void PanelDialog::moveEvent(QMoveEvent *event)
{
    QDialog::moveEvent(event);
}

void PanelDialog::closeEvent(QCloseEvent *event)
{
    if (shown)
        gm = geometry();

    if (getSettingsKey().length() > 0) {
        QSize size(gm.width(), gm.height());
        MW->settings->setValue(getSettingsKey(), size);
    }

    QDialog::closeEvent(event);
}

int PanelDialog::getDefaultWidth()
{
    return defaultWidth;
}

int PanelDialog::getDefaultHeight()
{
    return defaultHeight;
}

const QString PanelDialog::getSettingsKey()
{
    return settingsKey;
}
