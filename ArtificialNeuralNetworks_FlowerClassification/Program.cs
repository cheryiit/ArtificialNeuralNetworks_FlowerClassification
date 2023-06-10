using System.Globalization;

public class ArtificialNeuralNetwork
{
    public Neuron[] neurons;
    public int epoch;
    public double lambda;


    public ArtificialNeuralNetwork(Neuron[] neurons, int epoch, double lambda)
    {
        this.neurons = neurons;
        this.epoch = epoch;
        this.lambda = lambda;
    }

    public ArtificialNeuralNetwork(Neuron[] neurons)
    {
        this.neurons = neurons;
    }

    public void SetANN( int epoch, double lambda)
    {
        this.epoch = epoch;
        this.lambda = lambda;
    }
}

public class Neuron
{
    public double w1;
    public double w2;
    public double w3;
    public double w4;
    public int target;
    public string flowerOfInterest;

    public Neuron(string flowerOfInterest)
    {
        this.flowerOfInterest = flowerOfInterest;
    }

    public void SetWeights(double w1, double w2, double w3, double w4)
    {
        this.w1 = w1;
        this.w2 = w2;
        this.w3 = w3;
        this.w4 = w4;
    }

    public void SetTarget(int target)
    {
        this.target = target;
    }

    
}

public class Program
{
    private static int NUMBEROFSPLITDATA = 5;

    public static void Main(string[] args)
    {

        int numberOfNeurons = 3;
        Neuron[] neurons = new Neuron[numberOfNeurons];
        string[] flowerNames = { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };
        string fileName = Path.GetDirectoryName(System.AppDomain.CurrentDomain.BaseDirectory) + "\\iris.data";
        for (int i = 0; i < flowerNames.Length; i++)
        {
            neurons[i] = new Neuron(flowerNames[i]);
        }

        int[] EPECHarr = {20, 50, 100 };
        double[] lambdaArr = {0.01, 0.005, 0.025};
        ArtificialNeuralNetwork ANN = new ArtificialNeuralNetwork(neurons);
        string epochText;
        Console.WriteLine("                            Artificial-Neural-Network\n-----------------------------------IrisProject------------------------------------\n");
        Console.Write("      Lambda               ");
        for (int i = 0; i < lambdaArr.Length; i++) Console.Write(lambdaArr[i] + "                          ") ;
        Console.WriteLine("\nEpoch");
        for (int i = 0; i < lambdaArr.Length; i++)
        {
            epochText = ("" + EPECHarr[i]);
            Console.Write(epochText);
            for (int m = 0; m < 15 - epochText.Length; m++) Console.Write(" ");


            for (int j = 0; j < EPECHarr.Length; j++)
            {
                
                ANN.SetANN(EPECHarr[j], lambdaArr[i]);
                TrainingNetworkFromData(ANN, flowerNames, fileName);

                string text = testANN(fileName, neurons);
                Console.Write(text);
                for (int k = 0; k < 30 - text.Length; k++) Console.Write(" ");
                

            }
            Console.WriteLine();
        }
        Console.WriteLine("-----------------------------------------------------------------------------------\nPress enter for exit.");
        Console.ReadLine();
    }

    public static void TrainingNetworkFromData(ArtificialNeuralNetwork ANN, string[] flowerNames, string fileName)
    {
        string[] strDatas = new string[NUMBEROFSPLITDATA];
        double[] dataDoubles = new double[NUMBEROFSPLITDATA - 1];
        string flowerName;
        Neuron[] neurons = ANN.neurons;
        SetFirstWeights(neurons);
        string line;

        Neuron neuronTarget;
        Neuron maxNeuron;
         for (int timesOfOpech = 0; timesOfOpech < ANN.epoch ; timesOfOpech++)
         {
            using (StreamReader reader = new StreamReader(fileName))
            {
                
                while ((line = reader.ReadLine()) != null )
                {

                    strDatas = LineToData(line);
                    dataDoubles = GetNormalizeDoublesFromLineStr(strDatas);
                    flowerName = strDatas[NUMBEROFSPLITDATA - 1];
                    neuronTarget = setTargetFromFlowerName(neurons, flowerName);
                    maxNeuron = GetMaxOutputNeuron(neurons, dataDoubles);
                    if (maxNeuron.target != 1)
                    {
                        DecreaseWeights(maxNeuron, dataDoubles, ANN.lambda);
                        IncreaseWeights(neuronTarget, dataDoubles, ANN.lambda);
                    }
                    
                }
                reader.Close();

            }
         }
    }
    
    public static Neuron GetMaxOutputNeuron(Neuron[] neurons, double[] dataDoubles)
    {
        int maxIndex = 0;
        double maxOutput = -1;

        for (int i = 0; i < neurons.Length; i++)
        {
            if (maxOutput < CalculateNeuronOutput(neurons[i], dataDoubles))
            {
                maxOutput = CalculateNeuronOutput(neurons[i], dataDoubles);
                maxIndex = i;
            }

        }

        return neurons[maxIndex];
    }

    public static double CalculateNeuronOutput(Neuron n, double[] data)
    {
        double output = 0;
        double[] neuronWeights = {n.w1, n.w2, n.w3, n.w4};


        for (int i = 0; i < data.Length; i++)
        {
            output += neuronWeights[i] * data[i];
        }

        return output;
    }

    public static Neuron setTargetFromFlowerName(Neuron[] neurons, string flowerName)
    {
        int tempIndex = 0;
        for (int i = 0; i < neurons.Length; i++)
        {
            if (flowerName.Equals(neurons[i].flowerOfInterest))
            {
                neurons[i].SetTarget(1);
                tempIndex = i;
            }
            else
            {
                neurons[i].SetTarget(0);
            }
        }

        return neurons[tempIndex];
    }

    public static string[] LineToData(string line)
    {
        string[] dataArr = line.Split(',');

        return dataArr;
    }

    public static Neuron IncreaseWeights(Neuron n, double[] data, double lambda)
    {
        n.SetWeights(n.w1 + lambda * data[0], n.w2 + lambda * data[1], n.w3 + lambda * data[2], n.w4 + lambda * data[3]);
        
        return n;
    }

    public static Neuron DecreaseWeights(Neuron n, double[] data, double lambda)
    {
        n.SetWeights(n.w1 - lambda * data[0], n.w2 - lambda * data[1], n.w3 - lambda * data[2], n.w4 - lambda * data[3]);

        return n;
    }

    public static double NormalizationSigmoid(double data)
    {
        double normalizationInput = 1 / (1 + Math.Pow(Math.E, -1 * data));
        return normalizationInput;
    }

    public static double DeNormalizeSigmoid(double dataSigmoid)
    {
        double deNormalizedValue;
        double e = Math.E;
        deNormalizedValue = Math.Log(Math.Pow(dataSigmoid, -1) - 1) * -1;
        return deNormalizedValue;
    }

    public static double[] GetNormalizeDoublesFromLineStr(string[] data)
    {
        CultureInfo usCulture = new CultureInfo("en-US");
        NumberFormatInfo dataNumberFormat = usCulture.NumberFormat;

        double[] dataDoubles = new double[data.Length - 1];
        for (int i = 0; i < data.Length - 1; i++)
        {
            dataDoubles[i] = NormalizationSigmoid(double.Parse(data[i], dataNumberFormat)  );
            
        }
        return dataDoubles;
    }

    public static Neuron[] SetFirstWeights(Neuron[] neurons)
    {
        Random r = new Random();
        for (int i = 0; i < neurons.Length; i++)
        {
            neurons[i].w1 = r.NextDouble();
            neurons[i].w2 = r.NextDouble();
            neurons[i].w3 = r.NextDouble();
            neurons[i].w4 = r.NextDouble();
        }
        return neurons;
    }

    public static string testANN(string fileName, Neuron[] neurons )

    {
        string testString;
        using (StreamReader sr = new StreamReader(fileName))
        {
            double correctTests = 0;
            double inCorrectTests = 0;
            string[] strDatas;
            double[] dataDoubles;
            string flowerName;
            string line;
            while ((line = sr.ReadLine()) != null)
            {
                strDatas = LineToData(line);
                dataDoubles = GetNormalizeDoublesFromLineStr(strDatas);
                flowerName = strDatas[NUMBEROFSPLITDATA - 1];
                setTargetFromFlowerName(neurons, flowerName);
                if (GetMaxOutputNeuron(neurons, dataDoubles).target == 1)
                {
                    correctTests++;
                }
                else
                {
                    inCorrectTests++;
                }
            }
            sr.Close();
            double correctPercent = correctTests * 100.0 / (inCorrectTests + correctTests);
            testString =  "| Correct  = %" + String.Format("{0:0.000}", correctPercent) + " |";
        }

        return testString;
    }


}