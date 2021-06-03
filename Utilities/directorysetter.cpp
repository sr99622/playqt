/*******************************************************************************
* directorysetter.cpp
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

#include "directorysetter.h"

#include <QFileDialog>
#include <QGridLayout>

DirectorySetter::DirectorySetter(QMainWindow *parent, const QString& labelText)
{
    mainWindow = parent;
    label = new QLabel(labelText);
    text = new QLineEdit();
    button = new QPushButton("...");
    button->setMaximumWidth(40);
    connect(button, SIGNAL(clicked()), this, SLOT(selectDirectory()));

    QGridLayout *layout = new QGridLayout;
    layout->setContentsMargins(0, 0, 0, 0);
    if (label->text() != "")
        layout->addWidget(label,  0, 0, 1, 1);
    layout->addWidget(text,   0, 1, 1, 4);
    layout->addWidget(button, 0, 5, 1, 1);

    setLayout(layout);
}

void DirectorySetter::setPath(const QString& path)
{
    directory = path;
    text->setText(path);
}

void DirectorySetter::selectDirectory()
{
    QString path = QFileDialog::getExistingDirectory(mainWindow, label->text(), directory,
                                                  QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (path.length() > 0) {
        directory = path;
        text->setText(directory);
        emit directorySet(directory);
    }
}

void DirectorySetter::trimHeight()
{
    setMaximumHeight(label->fontMetrics().boundingRect("Xy").height() * 4);
}
