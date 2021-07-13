/*******************************************************************************
* messagedialog.h
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

#ifndef MESSAGEBOX_H
#define MESSAGEBOX_H

#include "paneldialog.h"
#include <QMainWindow>
#include <QTextEdit>

class MessageBox : public PanelDialog
{
    Q_OBJECT

public:
    MessageBox(QMainWindow *parent);
    int getDefaultWidth() override;
    int getDefaultHeight() override;
    QString getSettingsKey() const override;

    QTextEdit *message;

    const int defaultWidth = 400;
    const int defaultHeight = 400;
    const QString settingsKey = "MessageBox/size";

public slots:
    void clear();
    void copy();

};

#endif // MESSAGEBOX_H
