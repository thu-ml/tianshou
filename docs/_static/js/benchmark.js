var mujoco_envs = [
    "Ant-v4",
];

var atari_envs = [
    // "PongNoFrameskip-v4",
    // "BreakoutNoFrameskip-v4",
    // "EnduroNoFrameskip-v4",
    // "QbertNoFrameskip-v4",
    // "MsPacmanNoFrameskip-v4",
    // "SeaquestNoFrameskip-v4",
    // "SpaceInvadersNoFrameskip-v4",
];

function getDataSource(selectEnv, dirName) {
    return {
        // Paths are relative to the only file using this script, which is docs/01_tutorials/06_benchmark.rst
        $schema: "https://vega.github.io/schema/vega-lite/v5.json",
        data: {
            url: "../_static/js/" + dirName + "/benchmark/" + selectEnv + "/results.json"
        },
        mark: "line",
        height: 400,
        width: 800,
        params: [{name: "Range", value: 1000000, bind: {input: "range", min: 10000, max: 10000000}}],
        transform: [
            {calculate: "datum.iqm_confidence_interval[0]", as: "iqm_confidence_lower"},
            {calculate: "datum.iqm_confidence_interval[1]", as: "iqm_confidence_upper"},
            {calculate: "datum.iqm", as: "tooltip_str"},
            {filter: "datum.env_step <= Range"},
        ],
        encoding: {
            color: {"field": "agent", "type": "nominal"},
            x: {field: "env_step", type: "quantitative", title: "Env step"},
        },
        layer: [{
            "encoding": {
                "opacity": {"value": 0.3},
                "y": {
                    "field": "iqm_confidence_lower",
                    "type": "quantitative",
                },
                "y2": {"field": "iqm_confidence_upper"},
                tooltip: [
                    {field: "env_step", type: "quantitative", title: "Env step"},
                    {field: "agent", type: "nominal"},
                    {field: "tooltip_str", type: "nominal", title: "Return"},
                ]
            },
            "mark": "area"
        }, {
            "encoding": {
                "y": {
                    "title": "Return Interquartile Mean (5 seeds)",
                    "field": "iqm",
                    "type": "quantitative"
                }
            },
            "mark": "line"
        }]
    };
}

function showMujocoResults(elem) {
    const selectEnv = elem.value || mujoco_envs[0];
    const dataSource = getDataSource(selectEnv, "mujoco");
    vegaEmbed("#vis-mujoco", dataSource);
}

function showAtariResults(elem) {
    const selectEnv = elem.value || atari_envs[0];
    const dataSource = getDataSource(selectEnv, "atari");
    vegaEmbed("#vis-atari", dataSource);
}



document.addEventListener('DOMContentLoaded', function()  {
    var envMujocoSelect = $("#env-mujoco");
    if (envMujocoSelect.length) {
        $.each(mujoco_envs, function(idx, env) {envMujocoSelect.append($("<option></option>").val(env).html(env));})
        showMujocoResults(envMujocoSelect);
    }
    var envAtariSelect = $("#env-atari");
    if (envAtariSelect.length) {
        $.each(atari_envs, function(idx, env) {envAtariSelect.append($("<option></option>").val(env).html(env));})
        showAtariResults(envAtariSelect);
    }
});
