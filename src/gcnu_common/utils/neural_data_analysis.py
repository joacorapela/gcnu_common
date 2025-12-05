import numpy as np


def checkAtLeastOneSpikePerCluster(spikes_times):
    n_trials = len(spikes_times)
    n_clusters = len(spikes_times[0])
    for n in range(n_clusters):
        n_spikes = 0
        r = 0
        while n_spikes == 0 and r < n_trials:
            n_spikes = len(spikes_times[r][n])
            r += 1
        if n_spikes == 0:
            raise ValueError(f"cluster {n} has no spike across trials")


def checkSpikesTimesWithinBounds(spikes_times, trials_start_times,
                                 trials_end_times):
    n_trials = len(spikes_times)
    n_clusters = len(spikes_times[0])
    for r in range(n_trials):
        for n in range(n_clusters):
            if len(spikes_times[r][n]) > 0:
                min_spike_time_rn = min(spikes_times[r][n]) 
                if min_spike_time_rn < trials_start_times[r]:
                    raise ValueError(f"spike time at {min_spike_time_rn} found "
                                     f"before trial start time at "
                                     f"{trials_start_times[r]} for trials {r} "
                                     f"and cluster {n}")
                max_spike_time_rn = max(spikes_times[r][n]) 
                if max_spike_time_rn > trials_end_times[r]:
                    raise ValueError(f"spike time at {max_spike_time_rn} found "
                                     f"after trial start time at "
                                     f"{trials_start_times[r]} for trials {r} "
                                     f"and cluster {n}")


def checkEpochedSpikesTimes(spikes_times, trials_start_times, trials_end_times):
    checkAtLeastOneSpikePerCluster(spikes_times=spikes_times)
    checkSpikesTimesWithinBounds(spikes_times=spikes_times,
                                 trials_start_times=trials_start_times,
                                 trials_end_times=trials_end_times)


def getSpikesRatesAllTrialsAllClusters(spikes_times, trials_durations):
    n_trials = len(spikes_times)
    n_clusters = len(spikes_times[0])

    spikes_rates = np.empty((n_trials, n_clusters), dtype=np.double)
    for r in range(n_trials):
        for n in range(n_clusters):
            spikes_rates[r][n] = len(spikes_times[r][n])/trials_durations[r]
    return spikes_rates


# def binSpikes(spikes, bins_edges):
#     bin_counts, bins_edges_output = np.histogram(input=spikes, bins=bins_edges)
#     bin_centers = (bins_edges_output[:-1]+bins_edges_output[1:])/2.0
#     return bin_counts, bin_centers


def clipSpikesTimes(spikes_times, from_time, to_time):
    nTrials = len(spikes_times)
    clipped_spikes_times = [[]] * nTrials
    for r in range(nTrials):
        clipped_spikes_times[r] = clipTrialSpikesTimes(
            trial_spikes_times=spikes_times[r],
            from_time=from_time, to_time=to_time)
    return clipped_spikes_times


def clipTrialSpikesTimes(trial_spikes_times, from_time, to_time):
    nClusters = len(trial_spikes_times)
    clipped_trial_spikes_times = [[]] * nClusters
    for n in range(nClusters):
        clipped_trial_spikes_times[n] = clipClusterSpikesTimes(
            cluster_spikes_times=trial_spikes_times[n],
            from_time=from_time, to_time=to_time)
    return clipped_trial_spikes_times


def clipClusterSpikesTimes(cluster_spikes_times, from_time, to_time):
    clipped_cluster_spikes_times = cluster_spikes_times[
        np.logical_and(from_time <= cluster_spikes_times,
                       cluster_spikes_times < to_time)]
    return clipped_cluster_spikes_times


def offsetSpikeTimes(spikes_times, offset):
    nTrials = len(spikes_times)
    offsetted_spikes_times = [[]] * nTrials
    for r in range(nTrials):
        offsetted_spikes_times[r] = offsetTrialSpikesTimes(
            trial_spikes_times=spikes_times[r], offset=offset)
    return offsetted_spikes_times


def offsetTrialSpikesTimes(trial_spikes_times, offset):
    nClusters = len(trial_spikes_times)
    offsetted_trial_spikes_times = [[]] * nClusters
    for n in range(nClusters):
        offsetted_trial_spikes_times[n] = trial_spikes_times[n]+offset
    return offsetted_trial_spikes_times


def removeClusters(spikes_times, clusters_to_remove):
    nTrials = len(spikes_times)
    spikes_times_woClusters = [[]] * nTrials
    for r in range(nTrials):
        spikes_times_woClusters[r] = \
                removeClustersFromTrial(trial_spikes_times=spikes_times[r],
                                     clusters_to_remove=clusters_to_remove)
    return spikes_times_woClusters


def removeClustersFromTrial(trial_spikes_times, clusters_to_remove):
    nClusters = len(trial_spikes_times)
    spikes_times_woClusters = []
    for n in range(nClusters):
        if n not in clusters_to_remove:
            spikes_times_woClusters.append(trial_spikes_times[n])
    return spikes_times_woClusters


def selectClustersWithLessSpikesThanThrInAllTrials(spikes_times, thr):
    nTrials = len(spikes_times)
    nClusters = len(spikes_times[0])
    selected_clusters = set([i for i in range(nClusters)])
    for r in range(nTrials):
        selected_trial_clusters = selectClustersWithLessSpikesThanThrInTrial(
            spikes_times=spikes_times[r], thr=thr)
        selected_clusters = selected_clusters.intersection(selected_trial_clusters)
    answer = list(selected_clusters)
    return answer


def selectClustersWithLessSpikesThanThrInAnyTrial(spikes_times, thr):
    nTrials = len(spikes_times)
    selected_clusters = set([])
    for r in range(nTrials):
        selected_trial_clusters = selectClustersWithLessSpikesThanThrInTrial(
            spikes_times=spikes_times[r], thr=thr)
        selected_clusters = selected_clusters.union(selected_trial_clusters)
    answer = list(selected_clusters)
    return answer
    return selected_clusters


def selectClustersWithLessSpikesThanThrInTrial(spikes_times, thr):
    nClusters = len(spikes_times)
    selected_clusters = set([])
    for n in range(nClusters):
        if len(spikes_times[n]) < thr:
            selected_clusters.add(n)
    return selected_clusters


def removeClustersWithLessSpikesThanThrInAnyTrial(
        spikes_times, min_nSpikes_perCluster_perTrial):
    nClusters = len(spikes_times[0])
    clusters_indices = [n for n in range(nClusters)]
    clusters_to_remove = \
        selectClustersWithLessSpikesThanThrInAllTrials(
            spikes_times=spikes_times,
            thr=min_nSpikes_perCluster_perTrial)
    spikes_times = removeClusters(spikes_times=spikes_times,
                               clusters_to_remove=clusters_to_remove)
    clusters_indices = [n for n in clustens_indices
                       if n not in clusters_to_remove]
    return spikes_times, clusters_indices


def removeTrialsLongerThanThr(spikes_times, trials_indices,
                              trials_durations, max_trial_duration):
    trials_to_keep = np.where(trials_durations<=max_trial_duration)[0]
    spikes_times = [spikes_times[trial_to_keep]
                    for trial_to_keep in trials_to_keep]
    trials_indices = trials_indices[trials_to_keep]
    return spikes_times, trials_indices

def removeClustersWithLessTrialAveragedFiringRateThanThr(
        spikes_times, clusters_indices, trials_durations,
        min_cluster_trials_avg_firing_rate):
    n_clusters = len(spikes_times[0])
    n_trials = len(spikes_times)

    clusters_indices_to_keep = []
    for n in range(n_clusters):
        trials_firing_rates = np.array([np.nan for r in range(n_trials)])
        for r in range(n_trials):
            spikes_times_rn = spikes_times[r][n]
            trials_firing_rates[r] = len(spikes_times_rn)/trials_durations[r]
        trial_avg_firing_rate = trials_firing_rates.mean()
        if trial_avg_firing_rate > min_cluster_trials_avg_firing_rate:
            clusters_indices_to_keep.append(n)
    filtered_spikes_times = [[spikes_times[r][n]
                              for n in clusters_indices_to_keep]
                             for r in range(n_trials)]
    filtered_clusters_indices = [clusters_indices[n]
                                for n in clusters_indices_to_keep]
    return filtered_spikes_times, filtered_clusters_indices


def binClustersAndTrialsSpikesTimes(spikes_times, bins_edges, time_cluster):
    n_trials = len(spikes_times)
    n_clusters = len(spikes_times[0])
    binned_spikes_times = [[[] for n in range(n_clusters)]
                           for r in range(n_trials)]
    for r in range(n_trials):
        for n in range(n_clusters):
            binned_spikes_times[r][n] = binSpikesTimes(
                spikes_times=spikes_times[r][n],
                bins_edges=bins_edges,
                time_cluster=time_cluster)
    return binned_spikes_times

def binSpikesTimes(spikes_times, bins_edges, time_cluster):
    bin_width = bins_edges[1]-bins_edges[0]
    binned_spikes, _ = np.histogram(a=spikes_times, bins=bins_edges)
    binned_spikes = binned_spikes.astype(float)
    if time_cluster == "sec":
        binned_spikes *= 1.0/bin_width
    elif time_cluster == "msec":
        binned_spikes *= 1000.0/bin_width
    else:
        raise ValueError("time_cluster should be sec or msec, but not {}".format(time_cluster))
    return binned_spikes


def binMultiTrialSpikes(spikes_times, cluster_index, trials_indices,
                        bins_edges, time_cluster):
    mt_binned_spikes = np.empty((len(trials_indices), len(bins_edges)-1),
                                dtype=np.double)
    for i, trial_index in enumerate(trials_indices):
        aligned_spikes_trial_cluster = spikes_times[trial_index][cluster_index]
        binned_spikes = binSpikesTimes(
            spikes_times=aligned_spikes_trial_cluster,
            bins_edges=bins_edges, time_cluster=time_cluster)
        mt_binned_spikes[i, :] = binned_spikes
    return mt_binned_spikes


def computeBinnedSpikesAndPSTH(spikes_times, cluster_index, trials_indices,
                               bins_edges, time_cluster):
    binned_spikes = binMultiTrialSpikes(spikes_times=spikes_times,
                                        cluster_index=cluster_index,
                                        trials_indices=trials_indices,
                                        bins_edges=bins_edges,
                                        time_cluster=time_cluster)
    psth = np.empty(len(bins_edges)-1, dtype=np.double)
    for j in range(len(bins_edges)-1):
        psth[j] = binned_spikes[:, j].mean()
    return binned_spikes, psth


def computeBinnedSpikesAndPSTHwithCI(spikes_times, cluster_index,
                                     trials_indices, epoch_times,
                                     bins_edges, time_cluster,
                                     nResamples, alpha):
    binned_spikes = binMultiTrialSpikes(spikes_times=spikes_times,
                                        cluster_index=cluster_index,
                                        trials_indices=trials_indices,
                                        epoch_times=epoch_times,
                                        bins_edges=bins_edges,
                                        time_cluster=time_cluster)
    psth = np.empty(len(bins_edges)-1, dtype=np.double)
    psth_ci = np.empty((len(bins_edges)-1, 2), dtype=np.double)
    for j in range(len(bins_edges)-1):
        psth[j] = binned_spikes[:, j].mean()
        bootstrapped_mean = stats.bootstrapTests.bootstrapMean(
            observations=binned_spikes[:, j], nResamples=nResamples)
        psth_ci[j, :] = stats.bootstrapTests.estimatePercentileCI(
            alpha=alpha, bootstrapped_stats=bootstrapped_mean
        )
    return binned_spikes, psth, psth_ci


def alignAndClipSpikeTimes(spike_times, align_times, clip_start_time,
                           clip_end_time):
    nTrials = len(spike_times)
    nClusters = len(spike_times[0])
    aligned_clipped_spikes_times = []
    for r in range(nTrials):
        aligned_clipped_spikes_times_r = []
        for n in range(nClusters):
            aligned_spikes_times_rn = spike_times[r][n]-align_times[r]
            aligned_clipped_spikes_times_rn = aligned_spikes_times_rn[
                np.logical_and(clip_start_time <= aligned_spikes_times_rn,
                               aligned_spikes_times_rn < clip_end_time)
            ].tolist()
            aligned_clipped_spikes_times_r.append(
                aligned_clipped_spikes_times_rn)
        aligned_clipped_spikes_times.append(
            aligned_clipped_spikes_times_r)
    return aligned_clipped_spikes_times
