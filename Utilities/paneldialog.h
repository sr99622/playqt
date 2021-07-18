/*******************************************************************************
* paneldialog.h
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

#ifndef PANELDIALOG_H
#define PANELDIALOG_H

#include <QDialog>
#include <QMainWindow>
#include <QTimer>

#include "panel.h"
#include "guichangemonitor.h"

class PanelDialog : public QDialog
{
    Q_OBJECT

public:
    PanelDialog(QMainWindow *parent);
    void closeEvent(QCloseEvent *event) override;
    void showEvent(QShowEvent *event) override;
    void moveEvent(QMoveEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    virtual int getDefaultWidth();
    virtual int getDefaultHeight();
    virtual QString getSettingsKey() const;

    void saveSettings();

    QMainWindow *mainWindow;
    Panel *panel = nullptr;
    bool shown = false;
    bool changed = false;

    QTimer *timer;
    int defaultWidth = 320;
    int defaultHeight = 240;
    QString settingsKey = "";

signals:
    void closing();

public slots:
    void savePanelSettings();

};

#endif // PANELDIALOG_H
