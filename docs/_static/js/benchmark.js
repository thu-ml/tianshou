var mujoco_envs = [
    "Ant-v3",
    "HalfCheetah-v3",
    "Hopper-v3",
    "Humanoid-v3",
    "InvertedDoublePendulum-v2",
    "InvertedPendulum-v2",
    "Reacher-v2",
    "Swimmer-v3",
    "Walker2d-v3",
];

var atari_envs = [
    "PongNoFrameskip-v4",
    "BreakoutNoFrameskip-v4",
    "EnduroNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    "MsPacmanNoFrameskip-v4",
    "SeaquestNoFrameskip-v4",
    "SpaceInvadersNoFrameskip-v4",
];

function showMujocoEnv(elem) {
    var selectEnv = elem.value || mujoco_envs[0];
    var dataSource = {
        $schema: "https://vega.github.io/schema/vega-lite/v5.json",
        data: {
            // url: "/_static/js/mujoco/benchmark/" + selectEnv + "/result.json"
            url: "/en/master/_static/js/mujoco/benchmark/" + selectEnv + "/result.json"
        },
        mark: "line",
        height: 400,
        width: 800,
        params: [{name: "Range", value: 1000000, bind: {input: "range", min: 10000, max: 10000000}}],
        transform: [
            {calculate: "datum.rew - datum.rew_std", as: "rew_std0"},
            {calculate: "datum.rew + datum.rew_std", as: "rew_std1"},
            {calculate: "datum.rew + ' ± ' + datum.rew_std", as: "tooltip_str"},
            {filter: "datum.env_step <= Range"},
        ],
        encoding: {
            color: {"field": "Agent", "type": "nominal"},
            x: {field: "env_step", type: "quantitative", title: "Env step"},
        },
        layer: [{
            "encoding": {
                "opacity": {"value": 0.3},
                "y": {
                    "title": "Return",
                    "field": "rew_std0",
                    "type": "quantitative",
                },
                "y2": {"field": "rew_std1"},
                tooltip: [
                    {field: "env_step", type: "quantitative", title: "Env step"},
                    {field: "Agent", type: "nominal"},
                    {field: "tooltip_str", type: "nominal", title: "Return"},
                ]
            },
            "mark": "area"
        }, {
            "encoding": {
                "y": {
                    "field": "rew",
                    "type": "quantitative"
                }
            },
            "mark": "line"
        }]
    };
    vegaEmbed("#vis-mujoco", dataSource);
}

function showAtariEnv(elem) {
    var selectEnv = elem.value || atari_envs[0];
    var dataSource = {
        $schema: "https://vega.github.io/schema/vega-lite/v5.json",
        data: {
            // url: "/_static/js/atari/benchmark/" + selectEnv + "/result.json"
            url: "/en/master/_static/js/atari/benchmark/" + selectEnv + "/result.json"
        },
        mark: "line",
        height: 400,
        width: 800,
        params: [{name: "Range", value: 10000000, bind: {input: "range", min: 10000, max: 10000000}}],
        transform: [
            {calculate: "datum.rew - datum.rew_std", as: "rew_std0"},
            {calculate: "datum.rew + datum.rew_std", as: "rew_std1"},
            {calculate: "datum.rew + ' ± ' + datum.rew_std", as: "tooltip_str"},
            {filter: "datum.env_step <= Range"},
        ],
        encoding: {
            color: {"field": "Agent", "type": "nominal"},
            x: {field: "env_step", type: "quantitative", title: "Env step"},
        },
        layer: [{
            "encoding": {
                "opacity": {"value": 0.3},
                "y": {
                    "title": "Return",
                    "field": "rew_std0",
                    "type": "quantitative",
                },
                "y2": {"field": "rew_std1"},
                tooltip: [
                    {field: "env_step", type: "quantitative", title: "Env step"},
                    {field: "Agent", type: "nominal"},
                    {field: "tooltip_str", type: "nominal", title: "Return"},
                ]
            },
            "mark": "area"
        }, {
            "encoding": {
                "y": {
                    "field": "rew",
                    "type": "quantitative"
                }
            },
            "mark": "line"
        }]
    };
    vegaEmbed("#vis-atari", dataSource);
}

$(document).ready(function() {
    var envMujocoSelect = $("#env-mujoco");
    if (envMujocoSelect.length) {
        $.each(mujoco_envs, function(idx, env) {envMujocoSelect.append($("<option></option>").val(env).html(env));})
        showMujocoEnv(envMujocoSelect);
    }
    var envAtariSelect = $("#env-atari");
    if (envAtariSelect.length) {
        $.each(atari_envs, function(idx, env) {envAtariSelect.append($("<option></option>").val(env).html(env));})
        showAtariEnv(envAtariSelect);
    }
});
