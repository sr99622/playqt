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

void PanelDialog::showEvent(QShowEvent *event)
{
    shown = true;

    if (gm.width() == 0) {
        int cx = MW->geometry().center().x();
        int cy = MW->geometry().center().y();
        int dw = getDefaultWidth();
        int dh = getDefaultHeight();
        int dx = cx - dw/2;
        int dy = cy - dh/2;
        gm = QRect(dx, dy, dw, dh);
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
    Q_UNUSED(event);
    if (shown)
        gm = geometry();
    hide();
}

void PanelDialog::close()
{
    QCloseEvent event;
    closeEvent(&event);
}

int PanelDialog::getDefaultWidth()
{
    return defaultWidth;
}

int PanelDialog::getDefaultHeight()
{
    return defaultHeight;
}
