import { TRAINING_DATA } from './values.js';

// Извлечение тегов из разметки
let context = window.document.querySelector('canvas').getContext('2d');
let prediction = window.document.querySelector('#prediction');
let info = window.document.querySelector('#info');
let progress = window.document.querySelector('progress');

const counts = {};

// Функция для загрузки изображения в Canvas
const loadImageToCanvas = async (path) => {
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');

    const img = new Image();
    img.src = path;

    // load image
    await new Promise((resolve) => {
        img.onload = () => {
            // Отрисовка изображения на холсте
            context.drawImage(img, 0, 0, canvas.width, canvas.height);
            resolve(); // Резолвим промис после загрузки изображения
        };
    });
};

// Рисование изображения на основе числовых данных
// const drawImage = (digit) => {
//     let imageData = context.getImageData(0, 0, 28, 28);
//     for (let i = 0; i < digit.length; i++) {
//         imageData.data[i * 4] = digit[i] * 255; // red
//         imageData.data[i * 4 + 1] = digit[i] * 255; // green
//         imageData.data[i * 4 + 2] = digit[i] * 255; // blue
//         imageData.data[i * 4 + 3] = 255;
//     }
//     context.putImageData(imageData, 0, 0);
// };

// Создание стартовых тензоров
const INPUT_TENSOR = tf.tensor2d(TRAINING_DATA.inputs);
const OUTPUT_TENSOR = tf.oneHot(
    tf.tensor1d(TRAINING_DATA.outputs, 'int32'),
    10
);

// Создание модели
let model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [784], units: 32 }));
model.add(tf.layers.dense({ units: 16 }));
model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

// Обучение модели
const train = async () => {
    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });
    info.innerText = 'Training model. Please wait...';
    progress.style.display = 'block';
    const EPOCHS = 50;
    await model.fit(INPUT_TENSOR, OUTPUT_TENSOR, {
        batchSize: 512,
        epochs: EPOCHS,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                progress.value = (epoch / (EPOCHS - 1)) * 100;
                console.log(`Epoch ${epoch} `, logs);
            },
        },
    });
    info.innerText = 'Model succesfully trained';
    progress.style.display = 'none';
    // Удаление стартовых тензоров
    INPUT_TENSOR.dispose();
    OUTPUT_TENSOR.dispose();
};

await train();

// Тестирование модели
// const tryToPredict = () => {
//     let randomIndex = Math.floor(
//         Math.random() * TRAINING_DATA.inputs.length
//     );
//     let result = tf.tidy(() => {
//         let newInput = tf.tensor1d(TRAINING_DATA.inputs[randomIndex]);
//         let output = model.predict(newInput.expandDims());
//         return output.squeeze().argMax();
//     });
//     result.array().then((number) => {
//         prediction.innerText = number;
//         prediction.style.color =
//             number === TRAINING_DATA.outputs[randomIndex]
//                 ? '#00ff00' // green
//                 : '#ff0000'; // red
//         drawImage(TRAINING_DATA.inputs[randomIndex]);
//     });
// };
// tryToPredict();

// Функция для предсказания
const predict = async () => {
    // Получение данных изображения с холста
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = 'high';

    // ... ваш код для работы с контекстом canvas ...

    const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
    context.putImageData(imageData, 0, 0);

    // Предобработка изображения
    const inputTensor = tf.tidy(() => {
        const float32Array = new Float32Array(imageData.data);
        const normalized = tf.tensor(float32Array).div(255);

        // Изменение размерности тензора до двумерного
        const reshaped = normalized.reshape([28, 28, 4]);

        // Отбрасываем лишние каналы
        const grayscale = reshaped.slice([0, 0, 0], [-1, -1, 1]);

        // Изменение размера тензора с использованием resizeBilinear
        const resized = grayscale.resizeBilinear([28, 28]);

        const finalTensor = resized.reshape([1, 28 * 28]); // Изменение формы с использованием reshape
        return finalTensor;
    });

    // Предсказание
    const predictions = model.predict(inputTensor);
    const predictionArray = await predictions.data();

    // Определение предсказанной цифры
    const predictedDigit = predictionArray.indexOf(
        Math.max(...predictionArray)
    );

    //записываем значение в объект
    counts[predictedDigit] = (counts[predictedDigit] || 0) + 1;

    // Вывод результата
    prediction.innerText = `Predicted Digit: ${predictedDigit}`;
};

//generate pathFile
let str = '0000';
let pathFile = '';
for (let i = 0; i < 200; i++) {
    switch (i) {
        case 10:
            str = str.slice(1);
            break;
        case 100:
            str = str.slice(1);
            break;
        case 1000:
            str = str.slice(1);
            break;
        case 10000:
            0;
            str = str.slice(1);
            break;
    }

    pathFile = './digits/' + str + i + '.jpg';
    await loadImageToCanvas(pathFile);
    await predict();
}

const result = [];
for (let num in counts) {
    result[num] = `${num} встречается ${counts[num]} раз`;
}

console.log(result);
