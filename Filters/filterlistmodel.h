/*******************************************************************************
* filterlistmodel.h
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

#ifndef FILTERLISTMODEL_H
#define FILTERLISTMODEL_H

#include "filter.h"
#include <QAbstractListModel>
#include <QItemSelection>
#include <QVector>
#include <QWidget>
#include <QVBoxLayout>

class FilterListModel : public QAbstractListModel
{
    Q_OBJECT

public:
    FilterListModel();
    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int row = Qt::DisplayRole) const override;
    bool setData(const QModelIndex &index, const QVariant &value, int role = Qt::EditRole) override;
    Qt::ItemFlags flags(const QModelIndex &index) const override;

    QModelIndex setSelectedIndex(int row);
    QVector<Filter*> filters;
    int current_index = -1;
    //bool enablePanels = false;

signals:
    void panelShow(int index);
    //void panelHide(int index);

public slots:
    void onSelectedItemsChanged(QItemSelection selected, QItemSelection deselected);

private slots:
    void beginInsertItems(int start, int end);
    void endInsertItems();
};

#endif // FILTERLISTMODEL_H
