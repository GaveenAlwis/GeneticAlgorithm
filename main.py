#Evolutionary Computation Assignment 1: Group 10

#Instruction: To run the file type "python3 main.py". This will run it for all exercise's. If a specific exercise is wanted to be run, comment out all code in main that is not related to the specific exercise as labelled in the code. If a specific combination of selection, recombination and mutation is wanted to be run, type "geneticAlgorithm.run(20000, numberofPopulation, datasets[j],"{SELECTION}", "{RECOMBINATION}", "{MUTATION}") into main, in which:

#Selection:
#fitnessProportional= Fitness Proportional Selection 
#tournamentSelection= Tournament Selection 
#elitismSelection= Elitism Selection

#Crossover:
#edgeRecomb= Edge Recombination Selection 
#orderCrossover= Order Crossover
#PMXCrossover= PMX Crossover
#cycleCrossover= Cycle Crossover

#Mutation:
#scrambleMutation= Scramble Mutation
#insertMutation= Insert Mutation
#swapMutation= Swap Mutation
#inversionMutation= Inversion Mutation

## Import all neccesary libraries
import random
import numpy as np
import sys
## This limit was set as the current recursion limit is not sufficent for the large number of cities in some of the tsp files
sys.setrecursionlimit(14000)

# Exercise 2: 
## TSPProblem class reads in .tsp files and extract each cityid and corresponding x and y coordinates and saves all the information in an array
class TSPProblem:
    ## Intialises the return array and passes the file that needs to be extracted to the extractFile function.
    def __init__(self, documentName):
        self.city = np.array([])
        self.extractFile(documentName)
    ## Extract the city from the file.
    def extractFile(self, documentName):
        ## Open file and read
        with open(documentName, 'r') as file:
            lines = file.readlines()

        coordsCheck = None
        for i, line in enumerate(lines):
            if line.strip() == 'NODE_COORD_SECTION':
                coordsCheck = i+1
                break
        coordsLines = lines[coordsCheck:]
        city = []
        for line in coordsLines:
            # To find the end of file
            if line.strip() == 'EOF':
                break
            entireCoords = line.strip().split()
            if len(entireCoords) == 3:
                id, x, y = int(entireCoords[0]), float(
                    entireCoords[1]), float(entireCoords[2])
                city.append([id, x, y])
        self.city = np.array(city)
    ## Returns the cityid and corresponding x and y corrdinates upon request
    def returnCityData(self):
        return self.city


## Exercise 3
## Individual class forms a random permutation of the cities in a given TSP, in addition to calculating the total distances of traveling to all the cities of a given permutation and a method to override the permutation.
class Individual:
    ##Get a permutation of the id's by randomly shuffling in O(n)
    def __init__(self, num_cities):
        self.num_cities = num_cities
        self.permutation = np.arange(num_cities)
        np.random.shuffle(self.permutation)
    ## Returns the permutation
    def get_permutation(self):
        return self.permutation
    ## Connects the permutation to the id's and returns it
    def __str__(self):
        return ' '.join(str(city + 1) for city in self.permutation)
    ## Calculates the euclidean distance using the x and y coordinates at each city
    def distances(self, TSPCities):
        totaldistances = 0
        for i in range(self.num_cities-1):
            city1 = TSPCities[self.permutation[i]]
            city2 = TSPCities[self.permutation[i+1]]
            ycalc = city2[2]-city1[2]
            xcalc = city2[1]-city1[1]
            distance = np.sqrt(xcalc**2+ycalc**2)
            totaldistances += distance
        city1 = TSPCities[self.permutation[-1]]
        city2 = TSPCities[self.permutation[0]]
        ycalc = city2[2]-city1[2]
        xcalc = city2[1]-city1[1]
        distance = np.sqrt(xcalc**2+ycalc**2)
        totaldistances += distance
        return totaldistances
    ## Replaces the permutation of an individual with a new permutation
    def offspringIndividual(self, offspringPermutation):

        # self.permuation=offspringPermutation
        for i in range(0, self.num_cities):
            self.permutation[i] = offspringPermutation[i]

## Population class is a group of n individuals
class Population:
    ## Makes n amount of Individuals and put them in a array
    def __init__(self, num_cities, numIndividuals):
        self.totalPopulation = []
        for i in range(0, numIndividuals):
            self.totalPopulation.append(Individual(num_cities))
    ## Returns the Individuals array
    def returnPopulationIndividuals(self):
        return self.totalPopulation
    ## Return a list of all the distances for each individuals in the population
    def returnPopulationDistances(self, num_cities, TSP):
        populationDistances = []
        for i in range(0, len(self.totalPopulation)):
            populationDistances.append(self.totalPopulation[i].distances(TSP))
        return populationDistances
    ## A function to update the population with a new population
    def updatePopulation(self, index, IndividualObject):
        self.totalPopulation[index] = IndividualObject

## Excersise 4
## This class implements insert, swap, inversion and scramble mutations
class mutation:
    ## Establishes a local copy of genotype
    def __init__(self, genotype):
        self.genotype = genotype
    ## Implements insert mutation: Select a gene randomly and insert it into a random location
    def insertMutation(self, genotype):
        allele1 = np.random.randint(0, len(genotype))
        allele2 = np.random.randint(0, len(genotype))
        while allele1 == allele2:
            allele2 = np.random.randint(0, len(genotype))
        popPosition = self.genotype.pop(allele2)
        if allele1 < allele2:
            newAllelePosition = allele1+1

        else:
            newAllelePosition = allele1

        self.genotype.insert(newAllelePosition, popPosition)
        return genotype
    ## Implement swap mutations: Pick two data points at random and switch their positions
    def swapMutation(self, genotype):
        allele1, allele2 = random.sample(range(len(genotype)), 2)
        placeholder = genotype[allele1]
        genotype[allele1] = genotype[allele2]
        genotype[allele2] = placeholder
        return genotype
    ## Implement inversion mutation: A random subset of the array is selected and the order is reversed
    def inversionMutation(self, genotype):
        r1, r2 = sorted(random.sample(range(len(genotype)), 2))#create two  rand numbers
        while(r1==r2 or r1>r2):
            r1, r2 = sorted(random.sample(range(len(genotype)), 2))#create two  rand numbers


        b = genotype[r1:r2+1]
        b.reverse()

        iter=0
        for i in range(r1, r2 + 1):
            genotype[i] = b[iter]
            iter = iter+1
 
        return genotype
    
    ## Implement scamble mutation: A random subset of the array is selected and the order within that subset is randomized.
    def scrambleMutation(self, genotype):

        allele1 = np.random.randint(0, len(genotype))
        allele2 = np.random.randint(0, len(genotype))
 

        # how to odd case when aelle1 is equal to aelle2
        while (allele1 == allele2):
            allele1 = np.random.randint(0, len(genotype))
            allele2 = np.random.randint(0, len(genotype))

        # when the allele1  is less then allele2
        if allele1 > allele2:
            allele1, allele2 = allele2, allele1

        sublist = genotype[allele1:allele2]
        random.shuffle(sublist)


        new_genotype = genotype[:allele1] + sublist + genotype[allele2:]

        return new_genotype

## Implement Order, PMX, Cycle and edge recombination crossover
class crossover:
    ## Makes a local copy of genotype 1 and 2
    def __init__(self, genotype1, genotype2):
        self.genotype1 = genotype1
        self.genotype2 = genotype2
    ## Implementation of order crossover: Randomly selects a subset of the genomes and directly copies it into a random section of the offspring
    def orderCrossover(self, lst1: list, lst2: list):
        crossedLst = [-1]*len(lst1)
        allele1 = np.random.randint(0, len(lst1))
        allele2 = np.random.randint(0, len(lst1))
        mini = min(allele1, allele2)
        maxi = max(allele1, allele2)
        crossedLst[mini:maxi+1] = lst1[mini:maxi+1]
        idxParent2 = maxi+1
        idxCrossed = maxi+1
        while crossedLst.count(-1) > 0:
            if idxParent2 == len(crossedLst):
                idxParent2 = 0
            if idxCrossed == len(crossedLst):
                idxCrossed = 0
            if lst2[idxParent2] not in crossedLst:
                crossedLst[idxCrossed] = lst2[idxParent2]
                idxParent2, idxCrossed = idxParent2+1, idxCrossed+1
            else:
                idxParent2 += 1
        return crossedLst
    ## Implements PMX crossover: Take a subset of each parent at random and paste it into the corresponding position of the child. Then run mapping to ensure that no duplicates are present and fill the rest of the numbers at random
    def PMXCrossover(self, lst1: list, lst2: list):
        def outidx(idx: int, crossed: list, list2Range: list, list2: list):
            if crossed[idx] not in list2Range:
                return list2.index(crossed[idx])
            return outidx(list2Range.index(crossed[idx]), crossed, list2Range, list2)
        crossedLst = [-1]*len(lst1)
        allele1 = np.random.randint(0, len(lst1))
        allele2 = np.random.randint(0, len(lst1))
        mini = min(allele1, allele2)
        maxi = max(allele1, allele2)
        crossedLst[mini:maxi+1] = lst1[mini:maxi+1]
        for i in range(0, (maxi+1)-mini):
            if lst2[mini:maxi+1][i] not in crossedLst[mini:maxi+1]:
                crossedLst[outidx(i, crossedLst[mini:maxi+1],
                                  lst2[mini:maxi+1], lst2)] = lst2[mini:maxi+1][i]
            else:
                continue
        for index, element in enumerate(lst2):
            if element in crossedLst:
                continue
            else:
                crossedLst[index] = element
        return crossedLst
    ## Implements cycle crossover: Iterate through one parent, identifying and mapping the position in relation to the second parent. Repeat this until a cycle is formed. Use this cycle to form the child. 
    def cycleCrossover(self, lst1: list, lst2: list):
        def cycle(lst1: list, lst2: list, idx: int):
            start = lst1[idx]
            cycle = [start]
            while lst2[idx] != start:
                val = lst2[idx]
                idx = lst1.index(val)
                cycle.append(lst1[idx])
            return cycle
        cycles = []
        for i in range(len(lst1)):
            if lst1[i] in [item for sublist in cycles for item in sublist]:
                continue
            else:
                cycles.append(cycle(lst1, lst2, i))

        for k in range(1, len(cycles), 2):
            element = cycles[k]
            for i in range(len(lst1)):
                if lst1[i] not in element:
                    continue
                else:
                    temp = lst1[i]
                    lst1[i] = lst2[i]
                    lst2[i] = temp
        randVar=np.random.choice([0,1])
        
        if(randVar==0):
            result= lst1
        else:
            result=lst2

        result=np.array(result,dtype=np.int64)

        return result
    ## Implements edge recombination: Iterate through one parent, and write down both adjacent values at a given iterator on both parents, removing duplicates at the end. Then choose and node at random, then the next node would be the node that is adjacent to the first one and has the least number of neighbors. This step would be repeated until the child is full.
    def edgeRecomb(self, lst1: list, lst2: list):
        adjlist = {}
        ## Recursion loop
        def recursion(alleleList: list, finalList: list, dictionary: dict, allele: int):
            # print(len(dictionary))
            if len(dictionary) == 0:
                return finalList
            # print(type(alleleList))
            min_val_length = len(set(dictionary[alleleList[0]]))
            minList = [alleleList[0]]
            for element in set(alleleList):
                if dictionary[element].count(allele) == 2:
                    minList = [element]
                    break
                if len(set(dictionary[element])) < min_val_length:
                    min_val_length = len(set(dictionary[element]))
                    minList = [element]
                    continue
                if len(set(dictionary[element])) == min_val_length:
                    minList.append(element)
                    continue
            for key, subset in dictionary.items():
                dictionary[key] = [x for x in subset if x != allele]
            if len(minList) > 1:
                allele = np.random.choice(minList)
            else:
                allele = minList[0]

            finalList.append(allele)
            alleleList = dictionary[allele]
            del (dictionary[allele])
            return recursion(alleleList, finalList, dictionary, allele)

        for i, element in enumerate(lst1):
            idx = np.where(lst2 == element)[0][0]
            if element not in adjlist:
                adjlist[element] = []
            if i == 0:
                adjlist[element] += [lst1[i+1], lst1[len(lst1)-1]]
                if idx == 0:
                    adjlist[element] += [lst2[idx+1], lst2[len(lst1)-1]]
                    continue
                if idx == (len(lst2)-1):
                    adjlist[element] += [lst2[0], lst2[idx-1]]
                    continue
                else:
                    adjlist[element] += [lst2[idx-1], lst2[idx+1]]
                    continue
            if i == len(lst1)-1:
                adjlist[element] += [lst1[i-1], lst1[0]]
                if idx == 0:
                    adjlist[element] += [lst2[idx+1], lst2[len(lst2)-1]]
                    continue
                if idx == (len(lst2)-1):
                    adjlist[element] += [lst2[0], lst2[idx-1]]
                    continue
                else:
                    adjlist[element] += [lst2[idx-1], lst2[idx+1]]
                    continue
            else:
                adjlist[element] += [lst1[i+1], lst1[i-1]]
                if idx == 0:
                    adjlist[element] += [lst2[idx+1], lst2[len(lst2)-1]]
                    continue
                if idx == (len(lst2)-1):
                    adjlist[element] += [lst2[0], lst2[idx-1]]
                    continue
                else:
                    adjlist[element] += [lst2[idx-1], lst2[idx+1]]
                    continue


        allele = np.random.choice(lst1)
        finalList = [allele]
        startList = adjlist[allele]
        del (adjlist[allele])
        finalList = recursion(startList, finalList, adjlist, allele)
        return finalList

## Implements fitness proportional, tournament and elitism selection
class selection:
    ## Implements fitness proportional selection:  Each individual is put into a roulette wheel and a random individual is chosen. The chances an individual will be selected and calculated by its fitness divided by the total fitness of the population.
    def fitnessProportional(population, popDistances):
        fitnessEvaluated = []
        totalFitness = 0
        for i in (popDistances):
            fitnessEvaluated.append(1/i)

        fitnessSum = 0
        for j in fitnessEvaluated:
            fitnessSum += j

        selectionProbabilities = []
        for k in (fitnessEvaluated):
            selectionProbabilities.append(k/fitnessSum)

        fitnessProportionalValue = random.choices(population.returnPopulationIndividuals(
        ), weights=selectionProbabilities, k=len(population.totalPopulation))

        return (fitnessProportionalValue)
    ## Implements tournament selection: When a subset of the population is chosen, and the lowest distance individual in a given subset is the one that is selected
    def tournamentSelection(population, popDistances, tournament_size):
        allele_indices = random.sample(
            range(len(population.returnPopulationIndividuals())), tournament_size)

        selected_distances = [popDistances[i] for i in allele_indices]
        minimum_distance_index = allele_indices[selected_distances.index(
            min(selected_distances))]

        # This function returns mimumum_distance individual (allele) within the population
        return population.returnPopulationIndividuals()[minimum_distance_index]
    ## Implements Elitism Selection: Elitism is when the top individual (lowest distance) is guaranteed to be passed on to future generations, and given to breed.
    def elitismSelection(population, popDistances):
        minimum_distance_index = popDistances.index(min(popDistances))

        # This fucntion only returns top 1 individual (minimum),
        # can be altered to select n best choices later when design algorithm (Ex.6)
        return population.returnPopulationIndividuals()[minimum_distance_index]

## This uses all the functions and classes built upon prior to build a genetic algorithm.
class geneticAlgorithm:
    def run(generation, populationSize, TSP, selectionMethod, crossoverMethod, mutationMethod):
        ## Intialize the number of cities
        cityLength = len(TSP)
        ## Intialize a population of a given size
        population = Population(cityLength, populationSize)
        ## Calls the distances for all the individuals
        populationDistances = population.returnPopulationDistances(
            populationSize, TSP)
        
        print("Shortest route before:", min(populationDistances))
        ## Try and Exception clause to print out results if Keyboard exception is called after 4 hours
        try:
            ## A for loop to repeat the evolution process to the specified number of generation
            for i in range(0, generation):
                ## To print out the number of generations every 100 generations
                if(i%100==0):
                    print(i)
                # -----------Parent Selection-----------------
                # Runs selection method on population
                if (selectionMethod == "fitnessProportional"):
                    fitnessproportion = selection.fitnessProportional(
                        population, populationDistances)[0]
                if (selectionMethod == "tournamentSelection"):
                    fitnessproportion = selection.tournamentSelection(
                        population, populationDistances, 7)
                if (selectionMethod == "elitismSelection"):
                    fitnessproportion = selection.elitismSelection(
                        population, populationDistances)

                # -------------Recombination------------------
                ## The second parent selection is done through a random selection 
                fitnessRandom = np.random.choice(
                    population.returnPopulationIndividuals()[0:])
                while (fitnessRandom != fitnessproportion):
                    fitnessRandom = np.random.choice(
                        population.returnPopulationIndividuals()[0:])
                ## Run crossover on both parents (breeding them together)
                OrderCrossover1 = crossover(fitnessproportion, fitnessRandom)
                if (crossoverMethod == "edgeRecomb"):
                    offspring = OrderCrossover1.edgeRecomb(
                        fitnessproportion.get_permutation(), fitnessRandom.get_permutation())
                if (crossoverMethod == "orderCrossover"):
                    offspring = OrderCrossover1.orderCrossover(
                        fitnessproportion.get_permutation(), fitnessRandom.get_permutation())
                if (crossoverMethod == "PMXCrossover"):
                    offspring = OrderCrossover1.PMXCrossover(
                        fitnessproportion.get_permutation(), fitnessRandom.get_permutation())
                if (crossoverMethod == "cycleCrossover"):
                    offspring = OrderCrossover1.cycleCrossover(
                        fitnessproportion.get_permutation(), fitnessRandom.get_permutation())

                # -------------Mutation------------------------
                ## Run mutation on the returning permuation from crossover to add variance to the genome
                mutate = mutation(offspring)
                if (mutationMethod == "scrambleMutation"):
                    offspringMutated = mutate.scrambleMutation(offspring)
                if (mutationMethod == "insertMutation"):
                    offspringMutated = mutate.insertMutation(offspring)
                if (mutationMethod == "swapMutation"):
                    offspringMutated = mutate.swapMutation(offspring)
                if (mutationMethod == "inversionMutation"):
                    offspringMutated = mutate.inversionMutation(offspring)


                # -----------Survivor Selection---------------
                ## Removes the worst permuation with the new permutation

                temporaryIndividual = Individual(cityLength)
                temporaryIndividual.offspringIndividual(offspringMutated)
                populationDistances = population.returnPopulationDistances(10, TSP)
                worstDistance = populationDistances.index(max(populationDistances))
                population.updatePopulation(worstDistance, temporaryIndividual)
                ## At 5000, 10000 generation or at the end of the total number of generation the average and standard deviation of the distance values in a population would be returned
                if(i==5000):
                    sum=0
                    for j in range(0,len(populationDistances)):
                        sum=sum+populationDistances[j]
                    average=sum/len(populationDistances)
                    
                    print("Average distance after", selectionMethod, ",", crossoverMethod," and", mutationMethod," for generation",i,"a population size",populationSize, average,"and standard deviation:",np.std(populationDistances), "\n")
                if(i==10000):
                    sum=0
                    for j in range(0,len(populationDistances)):
                        sum=sum+populationDistances[j]
                    average=sum/len(populationDistances)
                    
                    print("Average distance after", selectionMethod, ",", crossoverMethod," and", mutationMethod," for generation",i,"a population size",populationSize, average,"and standard deviation:",np.std(populationDistances), "\n")
            sum=0
            for j in range(0,len(populationDistances)):
                sum=sum+populationDistances[j]
            average=sum/len(populationDistances)
            print("Average distance after", selectionMethod, ",", crossoverMethod," and", mutationMethod," for generation",i+1,"a population size",populationSize, average,"and standard deviation:",np.std(populationDistances), "\n")
            return (min(populationDistances))
        ## Return average and standard deviation of the current population on Keyboard Interupt
        except KeyboardInterrupt:
            sum=0
            for j in range(0,len(populationDistances)):
                sum=sum+populationDistances[j]
            average=sum/len(populationDistances)
            print("Average distance after", selectionMethod, ",", crossoverMethod," and", mutationMethod," for generation",i+1,"a population size",populationSize, average,"and standard deviation:",np.std(populationDistances), "\n")
            return (min(populationDistances))
## Excersise 7
class inverover:
    def inver_over_algorithim(self, population, population_size, numofgenerations, probability, TSP):
        # random initialization of the population p
        # termination condition is number of generations? - pretty sure this is correct from my understanding
        generation = 0
        while generation < numofgenerations:
            for i in range (0, population_size): #0 to 1 for testing atm
                # randomly select a city 
                pop = population.returnPopulationIndividuals()
                individual = pop[i]
                permutation = list(individual.get_permutation())
                # print("the length of the permutation before is: ", len(permutation))
                city_index = random.randrange(len(permutation)) # index of the current city
                city = permutation[city_index]
                city2 = 0
                while True:
                    #Need to include a repeat loop here for a certain condition, I think it is when the city at city_index+1 or city_index-1 = city2 but unsure
                    if np.random.rand() <= probability:
                        # print(len(permutation[:city_index]+permutation[city_index+1:]))
                        city2 = np.random.choice(permutation[:city_index]+permutation[city_index+1:]), 
                    else:
                        perm2 = list(np.random.choice(pop[:i]+pop[i+1:]).get_permutation())
                        # print(perm2)
                        temp = perm2.index(city)
                        if temp == len(perm2)-1:
                            city2 = perm2[0]
                    if city_index == 0 or city_index == len(permutation)-1:
                        if city_index == 0:
                            if permutation[1] == city2 or permutation[len(permutation)-1] == city2:
                            #break out of loop - need to add repeat loop i think
                                break
                        if city_index == len(permutation)-1:
                            if permutation[city_index-1] == city2 or permutation[0] == city2:
                                break
                    else:
                        if permutation[city_index-1] == city2 or permutation[city_index+1] == city2:
                        #break out of loop - need to add repeat loop i think
                            break
                
                    #inversing 
                    #confused on this section of the algo, what if index of c' in permutation is before initial city c?? - do we have to specify that the random generated value has to be after our city?
                    # print(permutation, city2)
                    if permutation.index(city2) < city_index:
                        inverse_section = permutation[permutation.index(city2):city_index]
                        inverse_section = inverse_section[::-1]
                        permutation = permutation[:permutation.index(city2)] + inverse_section + permutation[city_index:]
                        # print("The length of the permutation is (top): ", len(permutation))
                        
                    else:
                        inverse_section = permutation[city_index+1:permutation.index(city2)+1]
                        inverse_section = inverse_section[::-1]
                        permutation = permutation[:city_index+1] + inverse_section + permutation[permutation.index(city2)+1:]
                        # print("The length of the permutation is (bottom): ", len(permutation))
                    city = city2
                newIndiv = Individual(len(permutation))
                newIndiv.offspringIndividual(permutation)
                if newIndiv.distances(TSP) <= individual.distances(TSP):
                    population.updatePopulation(i, newIndiv)
            if generation % 5000 == 0:
                print(generation, " generations done")
            generation+=1   
        
        populationDistances = population.returnPopulationDistances(50, TSP)
        sum=0
        for k in range(0,len(populationDistances)):
            sum=sum+populationDistances[k]
        average=sum/len(populationDistances)
        print("The current average for the current population is: ", average,  "and standard deviation: ", np.std(populationDistances))
        return population
def main():
    # ----------Intialisation-------------
    
    # Part 1
    TSP = TSPProblem('Instances/a280.tsp').returnCityData()
    ## Excersise 6 Part1:
    ## This runs all possible combinations of selection, crossover and mutation on a single testcase to figure out which performs the best
    allSelection=[]
    allCrossover=[]
    allMutation=[]
    allSelection=["fitnessProportional","tournamentSelection","elitismSelection"]
    allCrossover=["edgeRecomb","orderCrossover","PMXCrossover","cycleCrossover"]
    allMutation=["scrambleMutation","swapMutation","inversionMutation"]
    for l in range(0,3):
        for m in range(0,4):
            for n in range(0,3):
                if(m==3 and n==0): ##cycle crossover and scramble dont work together
                    continue
                if(m==3 and n==2): ##cycle crossover and inversion dont work together
                    continue
                geneticAlgorithm.run(5000, 10, TSP, allSelection[l], allCrossover[m], allMutation[n])

    
    ## From the earlier code, the three following alogorithms were chosen to be th most optimal, the reasons are explained in the report.       
    run2 = geneticAlgorithm.run(
        20000, 10, TSP, "elitismSelection", "orderCrossover", "inversionMutation")
    run3 = geneticAlgorithm.run(
        20000, 10, TSP, "elitismSelection", "edgeRecomb", "inversionMutation")
    run1 = geneticAlgorithm.run(
        20000, 10, TSP, "tournamentSelection", "orderCrossover", "inversionMutation")



    # Experiment 1
    ##Initalises all the tsp files
    EIL51 = TSPProblem('Instances/eil51.tsp').returnCityData()
    EIL76 = TSPProblem('Instances/eil76.tsp').returnCityData()
    EIL101 = TSPProblem('Instances/eil101.tsp').returnCityData()
    st70 = TSPProblem('Instances/st70.tsp').returnCityData()
    kroA100 = TSPProblem('Instances/kroA100.tsp').returnCityData()
    kroC100 = TSPProblem('Instances/kroC100.tsp').returnCityData()
    lin105 = TSPProblem('Instances/lin105.tsp').returnCityData()
    pcb442 = TSPProblem('Instances/pcb442.tsp').returnCityData()
    pr2392 = TSPProblem('Instances/pr2392.tsp').returnCityData()
    usa13509 = TSPProblem('Instances/usa13509.tsp').returnCityData()
    ## Saves all the tsp file data in an array
    datasets=[]
    datasets=[EIL51,EIL76,EIL101,st70,kroA100,kroC100,lin105,pcb442,pr2392,usa13509]
    ## Runs all the test files at 10,20,50 and 100 population sizes at 20000 generation of 1 of our genetic algorithms (elitism, order crossover and inversion mutation)

    for i in range(0,len(datasets)):
        print(i)
        for numberofPopulation in [10,20,50,100]:

            geneticAlgorithm.run(20000, numberofPopulation, datasets[i],"elitismSelection", "orderCrossover", "inversionMutation")
    
    
    ## Runs all the test files at 10,20,50 and 100 population sizes at 20000 generation of 1 of our genetic algorithms (elitism, edge recombination crossover and inversion mutation)
    for j in range(0,len(datasets)):
        print(j)
        for numberofPopulation in [10,20,50,100]:

            geneticAlgorithm.run(20000, numberofPopulation, datasets[j],"elitismSelection", "edgeRecomb", "inversionMutation")

    ## Runs all the test files at 10,20,50 and 100 population sizes at 20000 generation of 1 of our genetic algorithms (tournament selection, order crossover and inversion mutation)
    for k in range(0,len(datasets)):
        print(k)
        for numberofPopulation in [10,20,50,100]:
     
            geneticAlgorithm.run(20000, numberofPopulation, datasets[k],"tournamentSelection", "orderCrossover", "inversionMutation")
    
    ## Excersise 6: Experiment 2:
    ## From the tests conducted in experiment 1, the most optimal combination seems to be Elitism Selection, Order Crossover and Inversion Mutation, an explanation of why is presented in the report
    ## The following lines will run this combination on all TSP files, our final finding will be found in the excel sheet provided.
    for l in range(0,len(datasets)):
        geneticAlgorithm.run(20000, 50, datasets[l],"elitismSelection", "orderCrossover", "inversionMutation")
    


    ## Question 7: The implements the innerover method of genetic algorithm
    inver = inverover()
    dataset = kroA100
    city_length = len(dataset)
    population = Population(city_length, 50)
    averageLst = []
    try:
        for j in range(0, 30):
            populationDistances = (inver.inver_over_algorithim(population, 50, 20000, 0.5, dataset)).returnPopulationDistances(50, dataset)
            print("run done")
            sum=0
            for k in range(0,len(populationDistances)):
                sum=sum+populationDistances[j]
            average=sum/len(populationDistances)
            averageLst.append(average)
            inver = inverover()
        sum=0
        for k in range(0,len(averageLst)):
            sum=sum+averageLst[k]
        average=sum/len(averageLst)
        print("number of iterations:", len(averageLst))
        print("Average for ", str(dataset), "is: ", average, "and standard deviation: ", np.std(populationDistances))
    except KeyboardInterrupt:
        if len(averageLst) > 0:
            sum = 0
            for k in range(0,len(averageLst)):
                sum=sum+averageLst[k]
            average=sum/len(averageLst)
            print("number of iterations:", len(averageLst))
            print("The average of all the averages for", str(dataset) , "is: ", average, "and standard deviation: ", np.std(populationDistances))



if __name__ == "__main__":
    main()

