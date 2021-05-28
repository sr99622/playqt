/*******************************************************************************
* numbertextbox.cpp
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

#include "numbertextbox.h"
#include <QKeyEvent>

NumberTextBox::NumberTextBox()
{
    setAlignment(Qt::AlignRight);
}

int NumberTextBox::intValue() const
{
    return text().toInt();
}

float NumberTextBox::floatValue() const
{
    return text().toFloat();
}

void NumberTextBox::setIntValue(int value)
{
    setText(QString::number(value));
}

void NumberTextBox::setFloatValue(float value)
{
    setText(QString::number(value));
}

void NumberTextBox::keyPressEvent(QKeyEvent *event)
{
    switch (event->key()) {
    case Qt::Key_Minus:
    case Qt::Key_0:
    case Qt::Key_1:
    case Qt::Key_2:
    case Qt::Key_3:
    case Qt::Key_4:
    case Qt::Key_5:
    case Qt::Key_6:
    case Qt::Key_7:
    case Qt::Key_8:
    case Qt::Key_9:
    case Qt::Key_Delete:
    case Qt::Key_Backspace:
    case Qt::Key_Insert:
    case Qt::Key_Home:
    case Qt::Key_End:
    case Qt::Key_Left:
    case Qt::Key_Right:
    case Qt::Key_Return:
    case Qt::Key_Enter:
    case Qt::Key_Tab:
    case Qt::Key_Period:
        QLineEdit::keyPressEvent(event);
        break;
    }
}
