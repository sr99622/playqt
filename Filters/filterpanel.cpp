/*******************************************************************************
* filterpanel.cpp
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

#include "filterpanel.h"
#include "mainwindow.h"
#include <QGridLayout>

#include "Filters/subpicture.h"
#include "Filters/darknet.h"

FilterDialog::FilterDialog(QMainWindow *parent) : PanelDialog(parent)
{
    mainWindow = parent;
    setWindowTitle("Filter");
    panel = new FilterPanel(mainWindow);
    QVBoxLayout *layout = new QVBoxLayout;
    layout->addWidget(panel);
    setLayout(layout);
}

void FilterDialog::keyPressEvent(QKeyEvent *event)
{
    //const Qt::KeyboardModifiers mods = event->modifiers();
    //const int code = event->key();

    //cout << "keyPressEvent" << endl;

    Filter *filter = panel->getCurrentFilter();
    if (filter) {
        panel->getCurrentFilter()->keyPressEvent(event);
    }

    PanelDialog::keyPressEvent(event);

}

void FilterDialog::keyReleaseEvent(QKeyEvent *event)
{
    Filter *filter = panel->getCurrentFilter();
    if (filter) {
        filter->keyReleaseEvent(event);
    }
   PanelDialog::keyReleaseEvent(event);
}

int FilterDialog::getDefaultWidth()
{
    return defaultWidth;
}

int FilterDialog::getDefaultHeight()
{
    return defaultHeight;
}

QString FilterDialog::getSettingsKey() const
{
    return settingsKey;
}

FilterPanel::FilterPanel(QMainWindow *parent)
{
    mainWindow = parent;

    leftView = new FilterListView;
    leftModel = new FilterListModel;
    leftModel->enablePanels = true;
    leftView->setModel(leftModel);
    connect(leftModel, SIGNAL(panelShow(int)), this, SLOT(panelShow(int)));
    connect(leftModel, SIGNAL(panelHide(int)), this, SLOT(panelHide(int)));
    connect(leftView, SIGNAL(doubleClick(QMouseEvent*)), this, SLOT(moveRight()));

    rightView = new FilterListView;
    rightModel = new FilterListModel;
    addFilters();
    for (int i = 0; i < filters.size(); i++) {
        rightModel->filters.push_back(filters[i]);
    }
    rightView->setModel(rightModel);
    connect(rightView, SIGNAL(doubleClick(QMouseEvent*)), this, SLOT(moveLeft()));

    connect(leftView->selectionModel(), SIGNAL(selectionChanged(QItemSelection, QItemSelection)), leftModel, SLOT(onSelectedItemsChanged(QItemSelection, QItemSelection)));
    connect(rightView->selectionModel(), SIGNAL(selectionChanged(QItemSelection, QItemSelection)), rightModel, SLOT(onSelectedItemsChanged(QItemSelection, QItemSelection)));

    moveRightButton = new QPushButton(">");
    moveRightButton->setMaximumWidth(70);
    QWidget *rightButtonPanel = new QWidget;
    QHBoxLayout *rightButtonLayout = new QHBoxLayout;
    rightButtonLayout->setContentsMargins(0, 0, 0, 0);
    rightButtonLayout->addWidget(moveRightButton);
    rightButtonPanel->setLayout(rightButtonLayout);

    moveLeftButton = new QPushButton("<");
    moveLeftButton->setMaximumWidth(70);
    QWidget *leftButtonPanel = new QWidget;
    QHBoxLayout *leftButtonLayout = new QHBoxLayout;
    leftButtonLayout->setContentsMargins(0, 0, 0, 0);
    leftButtonLayout->addWidget(moveLeftButton);
    leftButtonPanel->setLayout(leftButtonLayout);

    moveUpButton = new QPushButton("^");
    moveUpButton->setMaximumWidth(30);
    moveUpButton->setMinimumHeight(70);
    moveDownButton = new QPushButton("v");
    moveDownButton->setMaximumWidth(30);
    moveDownButton->setMinimumHeight(70);
    QWidget *upDownButtonPanel = new QWidget;
    QVBoxLayout *upDownButtonLayout = new QVBoxLayout;
    upDownButtonLayout->setContentsMargins(0, 0, 0, 0);
    upDownButtonLayout->addWidget(moveUpButton);
    upDownButtonLayout->addWidget(moveDownButton);
    upDownButtonPanel->setLayout(upDownButtonLayout);

    connect(moveLeftButton, SIGNAL(clicked()), this, SLOT(moveLeft()));
    connect(moveRightButton, SIGNAL(clicked()), this, SLOT(moveRight()));
    connect(moveUpButton, SIGNAL(clicked()), this, SLOT(moveUp()));
    connect(moveDownButton, SIGNAL(clicked()), this, SLOT(moveDown()));

    QGridLayout *viewLayout = new QGridLayout;
    viewLayout->setAlignment(Qt::AlignCenter);
    QWidget *viewPanel = new QWidget;
    viewLayout->addWidget(rightButtonPanel,    0, 1, 1, 1);
    viewLayout->addWidget(leftButtonPanel,     0, 2, 1, 1);
    viewLayout->addWidget(upDownButtonPanel,   1, 0, 1, 1);
    viewLayout->addWidget(leftView,            1, 1, 1, 1);
    viewLayout->addWidget(rightView,           1, 2, 1, 1);
    viewPanel->setLayout(viewLayout);

    tabWidget = new QTabWidget;
    tabWidget->setMinimumHeight(240);
    bottomPanel = new QWidget;
    QVBoxLayout *bottomLayout = new QVBoxLayout;
    for (int i = 0; i < rightModel->filters.size(); i++) {
        bottomLayout->addWidget(rightModel->filters[i]->panel);
        rightModel->filters[i]->panel->hide();
    }
    bottomPanel->setLayout(bottomLayout);

    tabWidget->addTab(bottomPanel, "");

    engageFilter = new QCheckBox("Engage Filter");

    QGridLayout *layout = new QGridLayout;
    layout->addWidget(viewPanel,        0, 0, 1, 1);
    layout->addWidget(tabWidget,        1, 0, 1, 1);
    layout->addWidget(engageFilter,     2, 0, 1, 1);
    layout->setRowStretch(0, 10);
    setLayout(layout);
}

FilterPanel::~FilterPanel()
{
    while (filters.size() > 0) {
        filters.pop_back();
    }
}

Filter *FilterPanel::getCurrentFilter()
{
    Filter *result = nullptr;
    //QModelIndex index = leftView->currentIndex();
    //if (index.isValid()) {
    if (leftModel->current_index > -1)
        result = leftModel->filters[leftModel->current_index];
    //}
    return result;
}

void FilterPanel::addFilters()
{
    filters.push_back(new SubPicture(mainWindow));
    filters.push_back(new Darknet(mainWindow));
}

void FilterPanel::saveSettings(QSettings *settings)
{
    for (int i = 0; i < filters.size(); i++) {
        QString arg = "FilterPanel_activeFilter_" + QString::number(i);
        settings->remove(arg);
    }

    for (int i = 0; i < filters.size(); i++) {
        QString arg = "FilterPanel_activeFilter_" + QString::number(i);
        settings->setValue(arg, "");
    }

    for (int i = 0; i < leftModel->filters.size(); i++) {
        QString arg = "FilterPanel_activeFilter_" + QString::number(i);
        settings->setValue(arg, leftModel->filters[i]->name);
    }

    for (int i = 0; i < filters.size(); i++)
        filters[i]->saveSettings(settings);
}

void FilterPanel::restoreSettings(QSettings *settings)
{
    QStringList list;
    for (int i = 0; i < filters.size(); i++) {
        //filters[i]->restoreSettings(settings);
        QString arg = "FilterPanel_activeFilter_" + QString::number(i);
        QString tmp = settings->value(arg).toString();
        if (tmp.length() > 0)
            list.push_back(tmp);
    }

    for (int i = 0; i < list.size(); i++) {
        for (int j = 0; j < rightModel->filters.size(); j++) {
            if (list[i] == rightModel->filters[j]->name) {
                rightModel->current_index = j;
                moveLeft();
                break;
            }
        }
    }

    //for (int i = 0; i < filters.size(); i++)
    //    filters[i]->restoreSettings(settings);
}

void FilterPanel::initializeFilters()
{
    for (int i = 0; i < filters.size(); i++) {
        filters[i]->initialize();
    }
}

void FilterPanel::idle()
{
    while (leftModel->filters.size() > 0) {
        leftModel->current_index = 0;
        moveRight();
    }
}

void FilterPanel::moveLeft()
{
    if (rightModel->filters.size() > 0 && rightModel->current_index > -1) {
        Filter *result = rightModel->filters[rightModel->current_index];
        for (int i = 0; i < leftModel->filters.size(); i++) {
            leftModel->filters[i]->panel->hide();
        }
        leftModel->filters.push_back(result);
        result->panel->show();
        result->initialize();
        leftModel->emit dataChanged(QModelIndex(), QModelIndex());
        leftView->selectionModel()->select(leftModel->setSelectedIndex(leftModel->filters.size()-1), QItemSelectionModel::SelectCurrent);
        rightModel->filters.remove(rightModel->current_index);
        rightModel->emit dataChanged(QModelIndex(), QModelIndex());
        if (rightModel->current_index == rightModel->filters.size()) {
            rightModel->current_index--;
            rightView->selectionModel()->select(rightModel->setSelectedIndex(rightModel->current_index), QItemSelectionModel::SelectCurrent);
        }
    }
}

void FilterPanel::moveRight()
{
    if (leftModel->filters.size() > 0 && leftModel->current_index > -1) {
        Filter *result = leftModel->filters[leftModel->current_index];
        result->panel->hide();
        result->deactivate();
        rightModel->filters.push_back(result);
        rightModel->emit dataChanged(QModelIndex(), QModelIndex());
        leftModel->filters.remove(leftModel->current_index);
        leftModel->emit dataChanged(QModelIndex(), QModelIndex());
        if (leftModel->current_index == leftModel->filters.size()) {
            leftModel->current_index--;
            if (leftModel->current_index > -1) {
                leftView->selectionModel()->select(leftModel->setSelectedIndex(leftModel->current_index), QItemSelectionModel::SelectCurrent);
                leftModel->filters[leftModel->current_index]->panel->show();
            }
            else {
                leftModel->current_index = 0;
            }
        }
        else {
            if (leftModel->filters.size() == 1) {
                leftModel->filters[0]->panel->show();
            }
        }
    }
}

void FilterPanel::moveUp()
{
    if (leftModel->filters.size() > 1 && leftModel->current_index > -1) {
        if (leftModel->current_index > 0) {
            int next_index = leftModel->current_index - 1;
            leftModel->filters.move(leftModel->current_index, next_index);
            leftView->selectionModel()->select(leftModel->setSelectedIndex(next_index), QItemSelectionModel::SelectCurrent);
            leftModel->emit dataChanged(QModelIndex(), QModelIndex());
        }
    }
}

void FilterPanel::moveDown()
{
    if (leftModel->filters.size() > 1 && leftModel->current_index > -1) {
        if (leftModel->current_index < leftModel->filters.size()-1) {
            int next_index = leftModel->current_index + 1;
            leftModel->filters.move(leftModel->current_index, next_index);
            leftView->selectionModel()->select(leftModel->setSelectedIndex(next_index), QItemSelectionModel::SelectCurrent);
            leftModel->emit dataChanged(QModelIndex(), QModelIndex());
        }
    }
}

void FilterPanel::panelShow(int index)
{
    //leftModel->filters[index]->initialize();
    leftModel->filters[index]->panel->show();
    tabWidget->setTabText(0, leftModel->filters[index]->name);
}

void FilterPanel::panelHide(int index)
{
    leftModel->filters[index]->panel->hide();
    if (leftModel->filters.size() == 0) {
        tabWidget->setTabText(0, "");
    }
}

bool FilterPanel::isFilterActive(Filter *filter) {
    bool result = false;

    for (int i = 0; i < leftModel->filters.size(); i++) {
        if (leftModel->filters[i]->name == filter->name) {
            result = true;
            i = leftModel->filters.size();
        }
    }
    return result;
}

bool FilterPanel::isFilterActive(QString filter_name)
{
    bool result = false;
    for (int i = 0; i < leftModel->filters.size(); i++) {
        if (leftModel->filters[i]->name == filter_name) {
            result = true;
            i = leftModel->filters.size();
        }
    }
    return result;
}

Filter *FilterPanel::getFilterByName(QString filter_name)
{
    Filter *result = NULL;
    for (int i = 0; i < filters.size(); i++) {
        if (filters[i]->name == filter_name) {
            result = filters[i];
            break;
        }
    }
    return result;
}
