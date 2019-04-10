#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
Идея решения, записать все возможные варианты и выбрать из них лучщший
*/

//Анализирует все варианты проходов и записывает их в stat_way и stat_call
__global__ void AllCall(char* str, int* size_str, int* stop_call, int *stat_way, int *stat_call) {
	//Получаем id текущего треда.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int i = 1;
	stat_way[idx]= 0;
	stat_call[idx] = 1;
	int offset = 0;
	if (idx > 0) {
		for (int j = 0; j < idx; j++) {
			offset += size_str[j];
		}
	}

	while (str[offset + i] != '\n') {
		if (str[offset + i] != str[offset + i - 1]) {
			stat_call[idx] += 1;
			if (stat_call[idx] > *stop_call) {
				stat_way[idx] = i;
				return;
			}
		}
		i++;
	}
	stat_way[idx] = i;
	return;
}

//Эти две функции пробегают по статистике и выбирают лучший вариант так, чтобы он отвечал условию на максимальное количество проходов
//Рекурсивно пробегает по статистике
__device__ void recurs(int* analys_way, int* analys_call, int stop_call, int leng_data_size, int call, int way, int idx, int *res) {
	if (call >= stop_call) {
		if (way > res[0]) {
			res[0] = way;
		}
		return;
	}
	else {
		for (int i = idx; i < leng_data_size; i++) {
			recurs(analys_way, analys_call, stop_call, leng_data_size, call + analys_call[i], way + analys_way[i], idx+1, res);
		}
		if (way > res[0]) {
			res[0] = way;
		}
	}
	return;
}

//Вызывает рекурсию
//Нельзя сразу вызвать рекурсию GPU с CPU, для этого нужна эта функция
__global__ void Analys(int* analys_way, int* analys_call, int* res_analys, int* stop_call, int * leng_data_size) {
	//Получаем id текущего треда.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int *res = (int*)malloc(sizeof(int));
	res[0] = 0;
	recurs(analys_way, analys_call, *stop_call, *leng_data_size, 0, 0, idx, res);
	printf("%d\n", res[0]);
	res_analys[idx] = res[0];
	return;
}


__host__ int main() {
	FILE *file; //файл
	int stop_call; //Количество проходов после которых устает
	int all_sum = 0; //Количество всех элементов в файле
	int *data_size = (int*)malloc(sizeof(int)); //Массив длин строк сообщение
	char name[20]; //Имя файла
	char elem; //Сюда записываем по элементу из файла
	char *all_data = (char*)malloc(sizeof(char)); //Тут храним все сообщение

	//Получаем имя файла и усталость
	printf("Enter name fail: ");
	gets_s(name);
	printf("Enter name tired: ");
	scanf_s("%d", &stop_call);

	//проверяем открывается ли файл
	if ((file = fopen(name, "r")) == false) {
		printf("Error open fail");
		system("pause");
		return 0;
	}

	elem = fgetc(file);

	//Переменные используемые для параметров размера
	int size_data = 1; 
	int leng_size_data = 1; //В дальнейшем будет фигурировать эта, как количество строк в сообщение
	int now_size_data = 0;
	int now_size = 0;
	int last_now_size = 0;

	//Мультипликативно выделяем память для массива строк и массива длин строк
	while (elem != EOF) {
		if (size_data == now_size) {
			size_data *= 2;
			all_data = (char*)realloc(all_data, size_data * sizeof(char));
		}
		all_data[now_size++] = elem;

		if (elem == '\n') {
			if (leng_size_data == now_size_data) {
				leng_size_data *= 2;
				data_size = (int*)realloc(data_size, leng_size_data * sizeof(int));
			}

			data_size[now_size_data] = now_size - last_now_size;
			last_now_size = now_size;

			now_size_data++;
		}

		elem = fgetc(file);
	}
	leng_size_data = now_size_data;
	size_data = now_size;
	all_data[size_data] = 0;

	for (int i = 0; i < leng_size_data; i++) {
		all_sum += data_size[i]-1;
	}
	//Инициализируем значения GPU
	char* str; //Сообщение
	int* size_str; //Размер сообщения
	int* tired; //Усталость
	int* stat_way; //Статистика прочитанных символов по каждой строке 
	int* stat_call; //Статистика проходов в сообщение
	//Тут сохраним результаты статистики на CPU
	int* res_stat_way = (int*)malloc(sizeof(int)*leng_size_data); 
	int* res_stat_call = (int*)malloc(sizeof(int)*leng_size_data);
	

	//Выделяем память для на видеокарте
	cudaMalloc((void**)&str, sizeof(char) * size_data);
	cudaMalloc((void**)&size_str, sizeof(int) * leng_size_data);
	cudaMalloc((void**)&tired, sizeof(int));
	cudaMalloc((void**)&stat_way, sizeof(int) * leng_size_data);
	cudaMalloc((void**)&stat_call, sizeof(int) * leng_size_data);
	
	//Запишем на GPU
	cudaMemcpy(str, all_data, sizeof(char) * size_data, cudaMemcpyHostToDevice);
	cudaMemcpy(size_str, data_size, sizeof(int) * leng_size_data, cudaMemcpyHostToDevice);
	cudaMemcpy(tired, &stop_call, sizeof(int), cudaMemcpyHostToDevice);
	

	dim3 gridSize = dim3(1, 1, 1);    //Размер используемой сетки
	dim3 blockSize = dim3(leng_size_data, 1, 1); //Размер используемого блока

	//Выполняем вызов функции ядра
	AllCall <<<gridSize, blockSize >>> (str, size_str, tired, stat_way, stat_call);

	//Инициализируем и создаем переменную синхронизации потоков
	cudaEvent_t syncEvent; 
	
	cudaEventCreate(&syncEvent);    //Создаем event
	cudaEventRecord(syncEvent, 0);  //Записываем event
	cudaEventSynchronize(syncEvent);  //Синхронизируем event

	//Выгружаем данные на CPU
	cudaMemcpy(res_stat_call, stat_call, sizeof(int)*leng_size_data, cudaMemcpyDeviceToHost);
	cudaMemcpy(res_stat_way, stat_way, sizeof(int)*leng_size_data, cudaMemcpyDeviceToHost);

	for (int i = 0; i < leng_size_data; i++) {
		printf("call: %d |", res_stat_call[i]);
		printf("way: %d\n", res_stat_way[i]);
	}
	
	cudaEventDestroy(syncEvent);

	cudaFree(str);
	cudaFree(size_str);

	//Обрабатываем статистику на GPU
	int* res_analys;
	int* number_leng;

	//Сохраняем результат в CPU
	int* res_array = (int*)malloc(sizeof(int)*leng_size_data);

	cudaMalloc((void**)&res_analys, sizeof(int)* leng_size_data);
	cudaMalloc((void**)&number_leng, sizeof(int));

	cudaMemcpy(number_leng, &leng_size_data, sizeof(int), cudaMemcpyHostToDevice);

	Analys <<< gridSize, blockSize >>> (stat_way, stat_call,  res_analys,  tired, number_leng);

	cudaEventCreate(&syncEvent);    //Создаем event
	cudaEventRecord(syncEvent, 0);  //Записываем event
	cudaEventSynchronize(syncEvent);  //Синхронизируем event

	cudaMemcpy(res_array, res_analys, sizeof(int)*leng_size_data, cudaMemcpyDeviceToHost);

	int res = 0;
	if (stop_call > leng_size_data) {
		res = all_sum;
	}
	else {
		//Находим максимальное количество правильно считанных элементов
		for (int i = 0; i < leng_size_data; i++) {
			if (res_array[i] > res) {
				res = res_array[i];
			}
		}
	}
	//Вычитаем из всех элемнетов, то есть получаем количество неправильно прочитанных
	printf("FINAL RES: %d", all_sum - res);
	
	cudaFree(tired);
	cudaFree(stat_way);
	cudaFree(stat_call);
	cudaFree(res_analys);
	cudaFree(number_leng);
	return 0;
}
