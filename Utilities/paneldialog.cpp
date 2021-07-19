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

    /*
    QRect gm;

    if (MW->settings->contains(getSettingsKey())) {
        gm = MW->settings->value(getSettingsKey()).toRect();
    }
    else {
        int w = getDefaultWidth();
        int h = getDefaultHeight();
        int x = MW->geometry().center().x() - w/2;
        int y = MW->geometry().center().y() - h/2;
        gm = QRect(x, y, w, h);
    }
    setGeometry(gm);
    */

    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(autoSave()));
    timer->start(10000);
}

void PanelDialog::keyPressEvent(QKeyEvent *event)
{
    if (event->modifiers() & Qt::ControlModifier) {
        QAction action(getSettingsKey());
        action.setShortcut(QKeySequence(Qt::CTRL | event->key()));
        MW->menuAction(&action);
    }

    QDialog::keyPressEvent(event);
}

void PanelDialog::showEvent(QShowEvent *event)
{
    shown = true;

    int w = getDefaultWidth();
    int h = getDefaultHeight();
    int x = MW->geometry().center().x() - w/2;
    int y = MW->geometry().center().y() - h/2;

    cout << "getSettingsKey: " << getSettingsKey().toStdString() << endl;

    if (getSettingsKey().length() > 0) {
        if (MW->settings->contains(getSettingsKey())) {
            cout << "Mainwindow contains settings key" << endl;
            QRect rect = MW->settings->value(getSettingsKey()).toRect();
            w = rect.width();
            h = rect.height();
            x = rect.x();
            y = rect.y();
            cout << "x: " << x << " y: " << y << " w: " << w << " h: " << h << endl;
        }
    }

    setGeometry(QRect(x, y, w, h));

    QDialog::showEvent(event);
}

void PanelDialog::autoSave()
{
    if (changed && shown && getSettingsKey().length() > 0) {
        cout << "settings key: " << getSettingsKey().toStdString() << endl;
        MW->settings->setValue(getSettingsKey(), geometry());
        changed = false;
    }

    if (panel)
        panel->autoSave();

}

void PanelDialog::resizeEvent(QResizeEvent *event)
{
    if (isVisible())
        changed = true;
    QDialog::resizeEvent(event);
}

void PanelDialog::moveEvent(QMoveEvent *event)
{
    if (isVisible())
        changed = true;
    QDialog::moveEvent(event);
}

void PanelDialog::closeEvent(QCloseEvent *event)
{
    //cout << "PanelDialog::closeEvent" << endl;
    autoSave();
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

QString PanelDialog::getSettingsKey() const
{
    return settingsKey;
}
