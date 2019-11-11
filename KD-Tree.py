#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import time

# 原作:github.com/wenffe/python-KD-
# 对着认真学习了一波

class Node(object):
    # node
    def __init__(self,item = None,label = None,dim = None,parent = None,left_child = None,right_child = None):
        self.item = item
        self.label = label
        self.dim = dim
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child
class KDTree(object):
    def __init__(self,data_list,label_list):
        self.__length = 0
        self.__root = self.__buildTree(data_list,label_list)
    def __buildTree(self,data_list,label_list,parentNode = None):
        '''
            构建kd树
            时间复杂度 : O(n^2)
            data_list : data
            label_list : label
            parent : parent node
        '''
        dataArray = np.array(data_list)
        # m 是样本个数,n 是样本维数
        m,n = dataArray.shape  
        labelArray = np.array(label_list).reshape(m,1)
        # 没有样本的情况
        if (m == 0):
            return None
        # 最大方差分割法
        var_list = [np.var(dataArray[:,col]) for col in range(n)]
        # 找到分割的维
        split = var_list.index(max(var_list))
        # 按最大方法维升序排序
        max_idx_list = dataArray[:,split].argsort()
        # 找到中间节点
        mid_item_index = max_idx_list[m//2]
        # 到达了叶子
        if (m == 1):
            self.__length += 1
            return Node(dim=split,label=label_list[mid_item_index],item=dataArray[mid_item_index],parent=parentNode,left_child=None,right_child=None)
        # 创建一个节点
        node = Node(dim=split,label=label_list[mid_item_index],item=dataArray[mid_item_index],parent=parentNode,left_child=None,right_child=None)
        
        left_tree = dataArray[:m//2]
        left_label = labelArray[:m//2]
        left_child = self.__buildTree(left_tree,left_label,node)
        if (m == 2):
            right_child = None
        else:
            right_tree = dataArray[m//2+1:]
            right_label = labelArray[m//2+1:] 
            right_child = self.__buildTree(right_tree,right_label,node)
        node.left_child = left_child
        node.right_child = right_child
        self.__length += 1
        return node
    
    @property
    def length(self):
        return self.__length
    @property
    def root(self):
        return self.__root
    
    def _find_nearest_neighbour(self,item):
        itemArray = np.array(item)
        if (self.length == 0):
            return None
        node = self.__root
        if (self.length == 1):
            return node
        while True:
            cur_dim = node.dim
            if (item[cur_dim] == node.item[cur_dim]):
                return node
            elif (item[cur_dim] < node.item[cur_dim]):
                if (node.left_child == None):
                    return node
                node = node.left_child
            else:
                if (node.right_child == None):
                    return node
                node = node.right_child
    
    def transfer_dict(self,node):
        '''
        查看kd树结构
        :node:需要传入根结点对象
        :return: 字典嵌套格式的kd树,字典的key是self.item,其余项作为key的值，类似下面格式
        {(1,2,3):{
                'label':1,
                'dim':0,
                'left_child':{(2,3,4):{
                                     'label':1,
                                     'dim':1,
                                     'left_child':None,
                                     'right_child':None
                                    },
                'right_child':{(4,5,6):{
                                        'label':1,
                                        'dim':1,
                                        'left_child':None,
                                        'right_child':None
                                        }
                }
        '''
        if node == None:
            return None
        kd_dict = {}
        kd_dict[tuple(node.item)] = {}  # 将自身值作为key
        kd_dict[tuple(node.item)]['label'] = node.label[0]
        kd_dict[tuple(node.item)]['dim'] = node.dim
        kd_dict[tuple(node.item)]['parent'] = tuple(node.parent.item) if node.parent else None
        kd_dict[tuple(node.item)]['left_child'] = self.transfer_dict(node.left_child)
        kd_dict[tuple(node.item)]['right_child'] = self.transfer_dict(node.right_child)
        return kd_dict
    
    def transfer_list(self,node, kdList=[]):
        '''
        将kd树转化为列表嵌套字典的嵌套字典的列表输出
        :param node: 需要传入根结点
        :return: 返回嵌套字典的列表
        '''
        if node == None:
            return None
        element_dict = {}
        element_dict['item'] = tuple(node.item)
        element_dict['label'] = node.label[0]
        element_dict['dim'] = node.dim
        element_dict['parent'] = tuple(node.parent.item) if node.parent else None
        element_dict['left_child'] = tuple(node.left_child.item) if node.left_child else None
        element_dict['right_child'] = tuple(node.right_child.item) if node.right_child else None
        kdList.append(element_dict)
        self.transfer_list(node.left_child, kdList)
        self.transfer_list(node.right_child, kdList)
        return kdList

    def knn_algo(self, item, k=1):
        '''
        找到距离测试样本最近的前k个样本
        :param item: 测试样本
        :param k: knn算法参数，定义需要参考的最近点数量，一般为1-5
        :return: 返回前k个样本的最大分类标签
        '''
        if self.length <= k:
            label_dict = {}
            # 获取所有label的数量
            for element in self.transfer_list(self.root):
                if element['label'] in label_dict:
                    label_dict[element['label']] += 1
                else:
                    label_dict[element['label']] = 1
            sorted_label = sorted(label_dict.items(), key=lambda item:item[1],reverse=True)  # 给标签排序
            return sorted_label[0][0]

        item = np.array(item)
        node = self._find_nearest_neighbour(item)  # 找到最近的那个结点
        if node == None:  # 空树
            return None
        print('靠近点%s最近的叶结点为:%s'%(item, node.item))
        node_list = []
        distance = np.sqrt(sum((item-node.item)**2))  # 测试点与最近点之间的距离
        least_dis = distance
        # 返回上一个父结点，判断以测试点为圆心，distance为半径的圆是否与父结点分隔超平面相交，若相交，则说明父结点的另一个子树可能存在更近的点
        node_list.append([distance, tuple(node.item), node.label[0]])  # 需要将距离与结点一起保存起来

        # 若最近的结点不是叶结点，则说明，它还有左子树
        if node.left_child != None:
            left_child = node.left_child
            left_dis = np.sqrt(sum((item-left_child.item)**2))
            if k > len(node_list) or least_dis < least_dis:
                node_list.append([left_dis, tuple(left_child.item), left_child.label[0]])
                node_list.sort()  # 对结点列表按距离排序
                least_dis = node_list[-1][0] if k >= len(node_list) else node_list[k-1][0]
        # 回到父结点
        while True:
            if node == self.root:  # 已经回到kd树的根结点
                break
            parent = node.parent
            # 计算测试点与父结点的距离，与上面距离做比较
            par_dis = np.sqrt(sum((item-parent.item)**2))
            if k >len(node_list) or par_dis < least_dis:  # k大于结点数或者父结点距离小于结点列表中最大的距离
                node_list.append([par_dis, tuple(parent.item) , parent.label[0]])
                node_list.sort()  # 对结点列表按距离排序
                least_dis = node_list[-1][0] if k >= len(node_list) else node_list[k - 1][0]

            # 判断父结点的另一个子树与结点列表中最大的距离构成的圆是否有交集
            if k >len(node_list) or abs(item[parent.dim] - parent.item[parent.dim]) < least_dis :  # 说明父结点的另一个子树与圆有交集
                # 说明父结点的另一子树区域与圆有交集
                other_child = parent.left_child if parent.left_child != node else parent.right_child  # 找另一个子树
                # 测试点在该子结点超平面的左侧
                if other_child != None:
                    if item[parent.dim] - parent.item[parent.dim] <= 0:
                        self.left_search(item,other_child,node_list,k)
                    else:
                        self.right_search(item,other_child,node_list,k)  # 测试点在该子结点平面的右侧

            node = parent  # 否则继续返回上一层
        # 接下来取出前k个元素中最大的分类标签
        label_dict = {}
        node_list = node_list[:k]
        # 获取所有label的数量
        for element in node_list:
            if element[2] in label_dict:
                label_dict[element[2]] += 1
            else:
                label_dict[element[2]] = 1
        sorted_label = sorted(label_dict.items(), key=lambda item:item[1], reverse=True)  # 给标签排序
        return sorted_label[0][0],node_list

    def left_search(self, item, node, nodeList, k):
        '''
        按左中右顺序遍历子树结点，返回结点列表
        :param node: 子树结点
        :param item: 传入的测试样本
        :param nodeList: 结点列表
        :param k: 搜索比较的结点数量
        :return: 结点列表
        '''
        nodeList.sort()  # 对结点列表按距离排序
        least_dis = nodeList[-1][0] if k >= len(nodeList) else nodeList[k - 1][0]
        if node.left_child == None and node.right_child == None:  # 叶结点
            dis = np.sqrt(sum((item - node.item) ** 2))
            if k > len(nodeList) or dis < least_dis:
                nodeList.append([dis, tuple(node.item), node.label[0]])
            return
        self.left_search(item, node.left_child, nodeList, k)
        # 每次进行比较前都更新nodelist数据
        nodeList.sort()  # 对结点列表按距离排序
        least_dis = nodeList[-1][0] if k >= len(nodeList) else nodeList[k - 1][0]
        # 比较根结点
        dis = np.sqrt(sum((item-node.item)**2))
        if k > len(nodeList) or dis < least_dis:
            nodeList.append([dis, tuple(node.item), node.label[0]])
        # 右子树
        if k > len(nodeList) or abs(item[node.dim] - node.item[node.dim]) < least_dis: # 需要搜索右子树
            if node.right_child != None:
                self.left_search(item, node.right_child, nodeList, k)

        return nodeList
    def right_search(self,item, node, nodeList, k):
        '''
        按右根左顺序遍历子树结点
        :param item: 测试的样本点
        :param node: 子树结点
        :param nodeList: 结点列表
        :param k: 搜索比较的结点数量
        :return: 结点列表
        '''
        nodeList.sort()  # 对结点列表按距离排序
        least_dis = nodeList[-1][0] if k >= len(nodeList) else nodeList[k - 1][0]
        if node.left_child == None and node.right_child == None:  # 叶结点
            dis = np.sqrt(sum((item - node.item) ** 2))
            if k > len(nodeList) or dis < least_dis:
                nodeList.append([dis, tuple(node.item), node.label[0]])
            return
        if node.right_child != None:
            self.right_search(item, node.right_child, nodeList, k)

        nodeList.sort()  # 对结点列表按距离排序
        least_dis = nodeList[-1][0] if k >= len(nodeList) else nodeList[k - 1][0]
        # 比较根结点
        dis = np.sqrt(sum((item - node.item) ** 2))
        if k > len(nodeList) or dis < least_dis:
            nodeList.append([dis, tuple(node.item), node.label[0]])
        # 左子树
        if k > len(nodeList) or abs(item[node.dim] - node.item[node.dim]) < least_dis: # 需要搜索左子树
            self.right_search(item, node.left_child, nodeList, k)

        return nodeList
        
if __name__ == '__main__':
    t1 = time.time()
    # dataArray = np.array( [[19,2 ],[ 7,0],[13,5],[3,15],[3,4],[ 3, 2], [ 8,9],[ 9,3],[17,15 ], [11,11]])
    dataArray = np.random.randint(0,20,size=(1000,2))
    # print('dataArray',dataArray)
    # label = np.array([[ 0],[ 1],[0],[ 1],[ 1],[ 1],[ 0],[ 1],[ 1], [1]])
    label = np.random.randint(0,3,size=(1000,1))
    # print('data',np.hstack((dataArray,label)))
    kd_tree = KDTree(dataArray,label)
    # kd_dict = kd_tree.transfer_dict(kd_tree.root)
    # print('kd_dict:',kd_dict)
    # kd_list = kd_tree.transfer_list(kd_tree.root)
    # print('kd_list',kd_list)
    # for i in kd_list:
    #     print(i)
    # node = kd_tree.find_nearest_neighbour([12,7])
    # print('%s最近的叶结点:%s'%([12,7],node.item))
    t2 = time.time()
    node = [3,3]
    label, node_list = kd_tree.knn_algo(node,k=5)
    print("点%s的最接近的前k个点为:%s"%(node, node_list))
    print("点%s的标签:%s"%(node,label))
    t3 = time.time()
    print("创建树耗时: %s"%(t2-t1))
    print("搜索前k个最近邻点耗时: %s"%(t3-t2))
