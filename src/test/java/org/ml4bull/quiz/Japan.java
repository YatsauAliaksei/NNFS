package org.ml4bull.quiz;

public interface Japan {
    String japan = "Код проходит в рамках цикла по массиву и выбирает наименьший элемент, ко-\n" +
            "торый еще не добавлен в отсортированную часть списка, затем меняет его местами\n" +
            "с элементом в позиции i.\n" +
            "Основные шаги алгоритма представлены на рисунке 6.2. На верхнем фрагмен-\n" +
            "те виден исходный неотсортированный массив. На среднем первые три элемента\n" +
            "уже отсортированы (они обведены толстой линией), а алгоритм готовится по-\n" +
            "менять позицию следующего элемента. Он ищет неотсортированные элементы,\n" +
            "чтобы выявить среди них тот, который имеет наименьшее значение, — в данном\n" +
            "случае это 3. Найденное число перемещается в следующую неотсортированную\n" +
            "позицию. На нижнем фрагменте показан массив после того, как новый элемент\n" +
            "был добавлен в отсортированную часть. Цикл For приступает к следующему\n" +
            "элементу со значением 5.\n" +
            "Как и в случае сортировки вставкой, алгоритм располагает упорядоченные эле-\n" +
            "менты в исходном массиве, поэтому дополнительной памяти ему не требуется (кро-\n" +
            "ме нескольких переменных для контроля над циклами и перемещения элементов).\n" +
            "Если в массиве содержится N элементов, алгоритм изучает каждый из них. Для на-\n" +
            "чала он должен пересмотреть N – i еще не отсортированных элементов, чтобы найти\n" +
            "принадлежащий позиции i, а затем передвинуть сортируемый элемент на последнюю позицию за малое количестве шагов. Таким образом, для перемещения всех\n" +
            "элементов потребуется (N – 1) + (N – 2) + ... + 2 + 1 = (N 2 + N)/2 шагов. Это значит,\n" +
            "что алгоритм обладает временем работы O(N 2 ), как и алгоритм сортировки вставкой. Сортировка выбором также является достаточно быстрой для относительно\n" +
            "малых массивов (менее 10 000 элементов). Кроме того, она проста и если элементов\n" +
            "совсем немного (от 5 до 10), работает эффективнее, чем более сложные алгоритмы.\n" +
            "Пузырьковая сортировка\n" +
            "Пузырьковая сортировка предполагает следующее: если массив не отсортиро-\n" +
            "ван, любые два смежных элемента в нем находятся в неправильном положении.\n" +
            "Из-за этого алгоритм должен проходить по массиву несколько раз, меняя местами\n" +
            "все неправильные пары. В коде используется булевская переменная not_sorted, которая отслеживает\n" +
            "перемещение элементов при прохождении через массив. Пока она истинна, цикл\n" +
            "работает — ищет неправильные пары элементов и перестраивает их.\n" +
            "Иллюстрацией этого алгоритма может служить рисунок 6.3. Первый массив\n" +
            "выглядит по большей части отсортированным, но, пройдясь по нему, алгоритм\n" +
            "выявит, что пара 6 — 3 находится в неправильном положении (число 6 должно\n" +
            "следовать после 3). Код поменяет найденные элементы местами и получит вто-\n" +
            "рой массив, в котором в неправильном положении окажется пара 5 — 3. После ее\n" +
            "исправления образуется третий массив, где неверна пара 4 — 3. Поменяв ее эле-\n" +
            "менты местами, алгоритм сформирует четвертый массив, совершит еще один ко-\n" +
            "нечный проход и, не найдя пар, стоящих в неправильном положении, остановится. В пузырьковой сортировке неупорядоченный элемент 3 как бы медленно\n" +
            "«всплывает» на правильную позицию, отсюда и специфическое название метода.\n" +
            "Каждое прохождение через массив ставит на нужное место как минимум один эле-\n" +
            "мент. В массиве, приведенном на рисунке 6.3, при первом прохождении на правиль-\n" +
            "ной позиции оказывается число 6, при втором — 5, при третьем — 3 и 4.\n" +
            "Если предположить, что в массиве содержится N элементов и хотя бы один из\n" +
            "них занимает свое место в результате однократного пересмотра значений, то алго-\n" +
            "ритм может совершить не более N прохождений. (Все N понадобятся, когда массив\n" +
            "изначально отсортирован в обратном порядке.) Каждое такое прохождение вклю-\n" +
            "чает N шагов, отсюда общее время работы алгоритма — O(N 2 ).\n" +
            "Как и две предыдущие сортировки, пузырьковая является довольно медлен-\n" +
            "ной, но может показать приемлемую производительность в малых списках (менее1000 элементов). Она также быстрее, чем более сложные алгоритмы для очень ма-\n" +
            "лых списков (около 5 элементов).\n" +
            "Вы можете несколько усовершенствовать пузырьковую сортировку. В масси-\n" +
            "ве, приведенном на рисунке 6.3, элемент со значением 3 находится ниже своей ко-\n" +
            "нечной позиции. А что если он будет выше нее? Алгоритм определит, что элемент\n" +
            "не там, где ему положено находиться, и поменяет местами со следующим элемен-\n" +
            "том, затем снова переставит, обнаружив неверный распорядок. Так продолжится\n" +
            "вниз по списку до тех пор, пока не отыщется конечная позиция. Чередуя прохож-\n" +
            "дения вверх и вниз по массиву, вы можете ускорить работу алгоритма: первые бы-\n" +
            "стрее поставят на место те элементы, которые находятся слишком низко в списке,\n" +
            "вторые — те, что слишком высоко.\n" +
            "Еще один способ усовершенствования — выполнять несколько перестановок за\n" +
            "проход. Например, при движении вниз по массиву элемент (назовем его K) может\n" +
            "сменить свою позицию не один раз, прежде чем займет нужное место. Вы сэконо-\n" +
            "мите время, если не будете передвигать его по массиву, а сохраните во временной\n" +
            "переменной и, сместив другие элементы вверх, найдете целевую позицию для K,\n" +
            "вставите его туда и продолжите прохождение.\n" +
            "Предположим, что в нашем массиве содержится наибольший элемент L,\n" +
            "который стоит не на своем месте. Двигаясь вниз, алгоритм добирается до него\n" +
            "(возможно, совершая другие перестановки) и затем перемещает вниз по списку,\n" +
            "пока тот не достигнет конечной позиции. Во время последнего прохождения\n" +
            "по массиву ни один элемент не может встать после L, поскольку этот элемент\n" +
            "уже находится на нужном месте. Значит, алгоритм может остановиться, когда\n" +
            "достигнет элемента L.Если обобщить вышесказанное, получается, что алгоритм завершает про-\n" +
            "хождение через массив, добравшись до позиции последней перестановки, кото-\n" +
            "рую выполнил во время предыдущего прохождения. Таким образом, отследив\n" +
            "последние перестановки при движении вниз и вверх по массиву, вы можете\n" +
            "сократить путь.\n" +
            "Рассмотренные усовершенствования проиллюстрированы на рисунке 6.4.\n" +
            "На крайнем левом фрагменте видно, что во время первого прохождения вниз по\n" +
            "массиву алгоритм вставляет элемент 7 во временную переменную и меняет местами\n" +
            "с элементами 4, 5, 6 и 3. Другими словами, алгоритму не требуется хранить элемент 7\n" +
            "в массиве до тех пор, пока он не займет конечную позицию.\n" +
            "Разместив 7 должным образом, алгоритм продолжает движение по массиву и не\n" +
            "находит других элементов для перестановки. Теперь ему известно, что 7 и следу-\n" +
            "ющие за ним элементы стоят в своих конечных позициях и рассматривать их больше\n" +
            "не нужно. Если бы какой-нибудь элемент, расположенный ближе к верху массива,\n" +
            "оказался больше 7, то при первом же прохождении он переместился бы вниз, минуя 7.\n" +
            "На среднем фрагменте элементы на конечных позициях закрашены серым — во\n" +
            "время последующих прохождений они не нуждаются в проверке.\n" +
            "На крайнем правом фрагменте алгоритм совершает второе прохождение по мас-\n" +
            "сиву вверх, начиная с элемента, стоящего перед 7. В нашем случае это 3. Алгоритм\n" +
            "меняет его местами с элементами 6, 5 и 4, удерживая во временной переменной,\n" +
            "пока не поставит на конечную позицию. Теперь 3 и предшествующие ему элементы\n" +
            "находятся на своих местах — они закрашены серым цветом. Вы можете выстраивать кучу, добавляя по одной вершине за раз. Начните\n" +
            "с дерева, состоящего из одной вершины. Поскольку у него нет дочерних за-\n" +
            "писей, оно уже является кучей. Новый узел нужно добавлять в конец дерева.\n" +
            "Чтобы оно оставалось полностью бинарным, это должен быть правый край\n" +
            "нижнего уровня.\n" +
            "Теперь сравните новое значение со значением родительской записи. Если первое\n" +
            "больше, поменяйте их местами. Поскольку дерево изначально формировалось как\n" +
            "куча, текущее родительское значение должно превышать уже имеющееся дочернее\n" +
            "(если оно есть). Поменяв местами родительский и новый дочерний элементы, вы\n" +
            "сохраните нужное отношение в данном узле, но можете нарушить его в узле вы-\n" +
            "шестоящем. Поднимитесь на уровень вверх, оцените располагающуюся там роди-\n" +
            "тельскую запись и, если нужно, произведите замену. Продолжайте это действие до\n" +
            "тех пор, пока не дойдете до корневой записи. Так вы получите кучу.\n" +
            "На рисунке 6.8 показан процесс, в ходе которого к дереву, изображенному на\n" +
            "рисунке 6.7, добавляют новую запись 12. Обновленную кучу вы можете увидеть\n" +
            "на рисунке 6.9. Внутри цикла алгоритм рассчитывает индексы дочерних записей текущего\n" +
            "узла. Если они оба выпадают из дерева, то их значения приравниваются к родитель-\n" +
            "скому. В таком случае дальнейшее сравнение индекса происходит с самим собой.\n" +
            "А поскольку любое значение больше или равно самому себе, подобный подход\n" +
            "удовлетворяет свойству кучи и недостающий узел не заставляет алгоритм менять\n" +
            "значения местами.\n" +
            "После того как алгоритм рассчитает дочерние индексы, он проверяет, сохраня-\n" +
            "ется ли свойство кучи в текущем месте. Если да, то алгоритм покидает цикл While.\n" +
            "То же самое происходит, если нет обеих дочерних записей или в отсутствии одной\n" +
            "из них другая соответствует свойству кучи.\n" +
            "Если свойство кучи не выполняется, алгоритм устанавливает swap_child на\n" +
            "индекс дочерней записи, содержащий большее значение, и меняет местами значения\n" +
            "родительского и дочернего узлов. Затем он обновляет переменную index, чтобы\n" +
            "сдвинуть ее вниз к переставленному дочернему узлу, и продолжает спускаться по\n" +
            "дереву.\n" +
            "Осуществление пирамидальной сортировки\n" +
            "Теперь, когда вы знакомы с построением кучи и знаете, как с ней работать, осу-\n" +
            "ществить пирамидальную сортировку для вас не составит труда. Приведенный ниже\n" +
            "алгоритм строит кучу. Он несколько раз меняет ее первый и последний элементы\n" +
            "местами и перестраивает все дерево, исключая последний элемент. При каждом\n" +
            "прохождении один элемент удаляется из кучи и добавляется в конец массива, где\n" +
            "расположены элементы в отсортированном порядке Сложнее определить время работы алгоритма. Чтобы построить исходную кучу,\n" +
            "ему приходится прибавлять каждый элемент к растущей куче. Всякий раз он разме-\n" +
            "щает его в конце дерева и просматривает структуру данных до самого верха, чтобы\n" +
            "убедиться, что она является кучей. Поскольку дерево полностью бинарное, количе-\n" +
            "ство его уровней равно O(log N). Таким образом, передвижение элемента вверх по\n" +
            "дереву может занять максимум O(log N) шагов. Алгоритм осуществляет шаг, при\n" +
            "котором добавляется элемент и восстанавливается свойство кучи, N раз, поэтому\n" +
            "общее время построения исходной кучи составит O(N u log N).\n" +
            "Для завершения сортировки алгоритм удаляет каждый элемент из кучи, а за-\n" +
            "тем восстанавливает ее свойства. Для этого он меняет местами последний элемент\n" +
            "кучи и корневой узел, а затем передвигает новый корень вниз по дереву. Дерево\n" +
            "имеет высоту O(log N) уровней, поэтому процесс может занять O(log N) време-\n" +
            "ни. Алгоритм повторяет такой шаг N раз, отсюда общее количество нужных ша-\n" +
            "гов — O(N u log N).\n" +
            "Суммирование времени, необходимого для построения исходной кучи и окон-\n" +
            "чания сортировки, дает следующее: O(N u log N) + O(N u log N) = O(N u log N).\n" +
            "Пирамидальная сортировка представляет собой так называемую сортировку\n" +
            "на месте и не нуждается в дополнительной памяти. На ее основе хорошо видно,\n" +
            "как работают кучи и как происходит сохранение полного бинарного дерева в мас-\n" +
            "сиве. И хотя производительность O(N log N) высока для алгоритма, сортирующе-\n" +
            "го путем сравнений, далее пойдет речь об алгоритме, который показывает еще бо-\n" +
            "лее быстрые результаты. Если изначально в массиве содержится N элементов и они разделены четко по-\n" +
            "полам, дерево вызовов быстрой сортировки имеет высоту log N уровней. При каж-\n" +
            "дом вызове проверяются элементы в той части массива, которая подвергается\n" +
            "сортировке. Например, в массиве из четырех элементов будут исследованы все\n" +
            "четыре элемента с целью последующего разделения их значений.\n" +
            "Все элементы из исходного массива представлены на каждом уровне дерева. Та-\n" +
            "ким образом, любой уровень содержит N элементов. Если вы добавите те, которые\n" +
            "должен проверять каждый вызов быстрой сортировки на определенном уровне де-\n" +
            "рева, получится N элементов. Это значит, что вызовы быстрой сортировки потре-\n" +
            "буют N шагов. Поскольку дерево имеет высоту log N уровней и для каждого из них\n" +
            "нужно N шагов, общее время работы алгоритма составит O(N u log N).\n" +
            "Подобный алгоритм предполагает разделение массива на две равные части при\n" +
            "каждом шаге, что выглядит не очень правдоподобно. Однако в большинстве случаев\n" +
            "разделяющий элемент будет отстоять недалеко от середины — не точно по центру,\n" +
            "но и не с самого края. Например, на рисунке 6.10 на среднем фрагменте разделяю-\n" +
            "щий элемент 6 находится хотя и не ровно посередине, но близко к ней. В этом слу-\n" +
            "чае алгоритм быстрой сортировки все еще обладает временем работы O(N u log N). В худшем случае разделяющий элемент может оказаться меньше любого другого\n" +
            "в той части массива, которую делит, либо все они будут иметь равные значения.\n" +
            "Тогда ни один из элементов не перейдет в левую часть массива — все, кроме раз-\n" +
            "деляющего, окажутся в правой. Первый рекурсивный вызов вернется немедленно,\n" +
            "поскольку сортировка не понадобится, зато в ходе второго потребуется обработать\n" +
            "почти каждый элемент. Если первому вызову быстрой сортировки надо отсорти-\n" +
            "ровать N элементов, то рекурсивному — N – 1.\n" +
            "Если разделяющий элемент всегда меньше остальных в сортируемой части\n" +
            "массива, алгоритм вызывается для сортировки вначале N элементов, затем N – 1,\n" +
            "N – 2 и т. д. В таком случае дерево вызовов, изображенное на рисунке 6.11, является\n" +
            "очень тонким и имеет высоту N.\n" +
            "Вызовы быстрой сортировки на уровне i в дереве должны проверить N – i\n" +
            "элементов. Суммирование всех элементов, проверяемых на всех вызовах, даст\n" +
            "N + (N – 1) + (N – 2) + ... + 1 = N u (N + 1)/2, что равняется O(N 2 ). Таким образом,\n" +
            "в худшем случае время работы алгоритма составит O(N 2 ). Кроме вышесказанного,\n" +
            "следует обратить внимание на требуемый объем памяти. Он частично зависит от\n" +
            "метода, с помощью которого массив делится на части, а также от глубины рекурсии\n" +
            "алгоритма. Если последовательность рекурсивных вызовов слишком глубока, про-\n" +
            "грамма расходует стековое пространство и зависнет.\n" +
            "В примере с деревом, изображенным на рисунке 6.11, алгоритм быстрой сор-\n" +
            "тировки рекурсивно вызывает сам себя до глубины N. Таким образом, в ожидае-\n" +
            "мом случае стек вызова программы будет иметь глубину O(log N) уровней. Для\n" +
            "большинства компьютеров это не проблема. Даже если в массиве содержится\n" +
            "1 млрд элементов, в log N их всего 30, а стек вызова должен обладать возможностью";

}
