/*******************************************************************************
* numbertextbox.h
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

#ifndef NUMBERTEXTBOX_H
#define NUMBERTEXTBOX_H

#include <QLineEdit>

class NumberTextBox : public QLineEdit
{
    Q_OBJECT

public:
    NumberTextBox();
    int intValue() const;
    float floatValue() const;
    void setIntValue(int value);
    void setFloatValue(float value);

private:
    void keyPressEvent(QKeyEvent *event) override;

};

#endif // NUMBERTEXTBOX_H
